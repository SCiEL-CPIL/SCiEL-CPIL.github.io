"""Ensemble Square Root Filter for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute Lorenz-96 right-hand side for a single state vector."""
    n = len(x)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x[(i + 1) % n] - x[(i - 2) % n]) * x[(i - 1) % n] - x[i] + F
    return dx


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Square Root Filter (EnSRF) analysis update for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (10,)
        Noisy observations of the observed indices.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Compute ensemble mean and anomalies
    ens_mean = ensemble.mean(axis=0)
    anomalies = ensemble - ens_mean  # (n_members, n_vars)
    
    # Sample covariance of anomalies
    cov = (anomalies.T @ anomalies) / (n_members - 1)  # (n_vars, n_vars)
    
    # Observation-related submatrices
    # Covariance of observed variables
    cov_oo = cov[np.ix_(observed_indices, observed_indices)]  # (n_obs, n_obs)
    # Cross-covariance: unobserved vs observed
    unobs_indices = np.setdiff1d(np.arange(n_vars), observed_indices)
    cov_uo = cov[np.ix_(unobs_indices, observed_indices)]  # (n_unobs, n_obs)
    # Covariance of unobserved variables
    cov_uu = cov[np.ix_(unobs_indices, unobs_indices)]  # (n_unobs, n_unobs)
    
    # Innovation: observation minus ensemble mean at observed locations
    innovation = observations - ens_mean[observed_indices]  # (n_obs,)
    
    # Regularize observation covariance
    R = observation_variance * np.eye(n_obs)
    
    # Compute Kalman gain for unobserved variables
    # K_u = cov_uo @ inv(cov_oo + R)
    try:
        S = cov_oo + R  # (n_obs, n_obs)
        # Use Cholesky for numerical stability
        L = np.linalg.cholesky(S)
        # Solve L @ X = cov_uo.T  =>  X = L^{-1} cov_uo.T
        X = np.linalg.solve(L, cov_uo.T)  # (n_obs, n_unobs)
        # K_u = X @ L^{-1}  =>  solve L^T @ K_u^T = X^T
        K_u_T = np.linalg.solve(L.T, X.T)  # (n_unobs, n_obs)
        K_u = K_u_T.T  # (n_unobs, n_obs)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        try:
            K_u = cov_uo @ np.linalg.pinv(cov_oo + R)
        except:
            K_u = np.zeros((len(unobs_indices), n_obs))
    
    # Update unobserved variables
    update_u = K_u @ innovation  # (n_unobs,)
    
    # For observed variables, we also need the gain
    try:
        K_o = np.linalg.solve(S, cov_oo.T).T  # (n_obs, n_obs)
    except np.linalg.LinAlgError:
        try:
            K_o = np.linalg.pinv(cov_oo + R) @ cov_oo
        except:
            K_o = np.eye(n_obs) * 0.5
    
    update_o = K_o @ innovation  # (n_obs,)
    
    # Build full update vector
    update = np.zeros(n_vars)
    update[observed_indices] = update_o
    update[unobs_indices] = update_u
    
    # Apply deterministic update to all members (mean shift)
    result = ensemble.copy()
    result += update[np.newaxis, :]
    
    # Now apply the stochastic part: adjust anomalies
    # The analysis anomalies should be: a_a = (I - K H) a_f
    # where H is the observation operator (selects observed indices)
    # For observed variables: a_a[obs] = (I - K_o) a_f[obs]
    # For unobserved variables: a_a[unobs] = (I - K_u H) a_f[unobs] = a_f[unobs] - K_u a_f[obs]
    
    # Compute K H matrix for the full state
    KH = np.zeros((n_vars, n_vars))
    KH[np.ix_(observed_indices, observed_indices)] = K_o
    KH[np.ix_(unobs_indices, observed_indices)] = K_u
    
    # Update anomalies
    new_anomalies = anomalies - (KH @ anomalies.T).T  # (n_members, n_vars)
    
    # Reconstruct ensemble
    result = ens_mean + update + new_anomalies
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, ens_mean[np.newaxis, :])
    
    # Clip to [-100, 100]
    result = np.clip(result, -100.0, 100.0)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Should not happen for this simple update, but safety check
        pass
    
    return result

# EVOLVE-BLOCK-END
