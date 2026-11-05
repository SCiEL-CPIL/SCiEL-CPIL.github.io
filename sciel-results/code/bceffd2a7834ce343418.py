"""Lorenz-96 assimilation using Ensemble Kalman Filter with localization and adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Time deadline guard
_DEADLINE = 140.0
_t0 = None

def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an Ensemble Square Root Filter analysis update with sequential coordinate-wise regression.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Work with a copy
    ens = ensemble.astype(np.float64).copy()
    
    # Compute ensemble mean
    ens_mean = np.mean(ens, axis=0)
    
    # Observation error variance
    r_var = observation_variance
    
    # Compute ensemble anomalies
    anoms = ens - ens_mean  # (n_members, n_state)
    
    # Compute the regression coefficients for each state variable onto observations
    # Using the ensemble covariance structure
    
    # Observed anomalies
    obs_anoms = anoms[:, observed_indices]  # (n_members, n_obs)
    
    # Compute the "regression" of each state variable onto the observed variables
    # P_xo = Cov(x, y) = (1/(n-1)) * X_anom^T @ Y_anom
    # P_yy = Cov(y, y) = (1/(n-1)) * Y_anom^T @ Y_anom
    
    # For each state variable i, we want to regress it onto the observations
    # x_i_a = x_i_f + beta_i^T * (y - y_f)
    # where beta_i = P_xi_y * P_yy^{-1}
    
    # To avoid inverting P_yy (which is 10x10 but may be ill-conditioned with 16 members),
    # we use the Cholesky decomposition approach
    
    # Add regularization to P_yy
    P_yy = (obs_anoms.T @ obs_anoms) / (n_members - 1)  # (n_obs, n_obs)
    P_yy += 1e-6 * np.eye(n_obs)
    
    # Cholesky decomposition of P_yy
    try:
        L_yy = np.linalg.cholesky(P_yy)
    except np.linalg.LinAlgError:
        L_yy = np.linalg.cholesky(P_yy + 1e-4 * np.eye(n_obs))
    
    # For each state variable, compute the regression coefficients
    # beta_i = P_xi_y @ P_yy^{-1} = (X_i_anom^T @ Y_anom / (n-1)) @ P_yy^{-1}
    # Using Cholesky: P_yy^{-1} = L^{-T} @ L^{-1}
    
    # We can compute this more efficiently using the ensemble directly
    # The update in square root form: for each unobserved variable,
    # x_i_a = x_i_f + (x_i_anom - beta_i @ y_anom) * something
    
    # Actually, let's use the standard EnSRF formulation:
    # For each state variable i (both observed and unobserved):
    # 1. Compute the regression of x_i onto the observations
    # 2. Update using the innovation
    
    # First, let's do the observed variables directly (they should match observations)
    # Then propagate to unobserved variables via regression
    
    # Compute the Kalman gain-like matrix using the ensemble
    # K = P_xo @ (P_yy + R)^{-1}
    
    P_xo = (anoms.T @ obs_anoms) / (n_members - 1)  # (n_state, n_obs)
    
    S = P_yy + r_var * np.eye(n_obs)
    
    try:
        L_S = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        L_S = np.linalg.cholesky(S + 1e-4 * np.eye(n_obs))
    
    # K = P_xo @ S^{-1} = P_xo @ L_S^{-T} @ L_S^{-1}
    # Compute S^{-1} @ P_xo^T efficiently
    # Let Z = L_S^{-1} @ P_xo^T, then K^T = Z^T @ L_S^{-T}
    # Actually: K = P_xo @ S^{-1}
    # S^{-1} = L_S^{-T} @ L_S^{-1}
    # So K = P_xo @ L_S^{-T} @ L_S^{-1}
    
    # Compute M = P_xo @ L_S^{-T} by solving L_S^T @ M^T = P_xo^T
    M_T = np.linalg.solve(L_S.T, P_xo.T)  # (n_obs, n_state)
    M = M_T.T  # (n_state, n_obs)
    
    # Now K = M @ L_S^{-1}
    # Solve L_S @ K^T = M^T
    K_T = np.linalg.solve(L_S, M.T)  # (n_obs, n_state)
    K = K_T.T  # (n_state, n_obs)
    
    # Apply localization to the Kalman gain
    # Use a Gaussian decay based on circular distance
    decay_length = 2.0
    localized_K = np.zeros_like(K)
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        for i in range(n_state):
            d = abs(i - obs_idx)
            d = min(d, n_state - d)
            w = np.exp(-d ** 2 / (2.0 * decay_length ** 2))
            localized_K[i, j] = K[i, j] * w
    
    # Apply the update to the ensemble
    # For each member: x_a = x_f + K @ (y - H@x_f)
    Hx = ens[:, observed_indices]  # (n_members, n_obs)
    innovations = observations[None, :] - Hx  # (n_members, n_obs)
    
    update = innovations @ localized_K.T  # (n_members, n_state)
    ens = ens + update
    
    # Adaptive inflation using the ensemble spread
    ens_mean_new = np.mean(ens, axis=0)
    
    # Compute the expected innovation variance
    obs_ens_new = ens[:, observed_indices]
    obs_anoms_new = obs_ens_new - np.mean(obs_ens_new, axis=0)
    P_oo_new = (obs_anoms_new.T @ obs_anoms_new) / (n_members - 1)
    
    S_expected = P_oo_new + r_var * np.eye(n_obs)
    S_expected_diag = np.diag(S_expected)
    
    # Actual innovation variance
    innov_sq = (observations - ens_mean) ** 2
    actual_var = innov_sq + r_var
    
    # Compute inflation factor
    ratios = []
    for j in range(n_obs):
        exp_var = S_expected_diag[j]
        act_var = actual_var[j]
        if exp_var > 1e-8 and act_var > 1e-8:
            ratios.append(act_var / exp_var)
    
    if len(ratios) > 0:
        log_ratios = np.log(np.clip(ratios, 1e-4, 1e4))
        geo_mean_ratio = np.exp(np.mean(log_ratios))
        inflation = np.sqrt(np.clip(geo_mean_ratio, 0.9, 1.5))
        inflation = np.clip(inflation, 0.95, 1.30)
    else:
        inflation = 1.0
    
    # Apply inflation around the new mean
    ens = ens_mean_new + inflation * (ens - ens_mean_new)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
