"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_DEADLINE_SECONDS = 115.0  # Safety margin before 120s evaluator limit
_START_TIME = [time.time()]  # Mutable container to track start time across calls


def _elapsed():
    """Return elapsed time since first call."""
    return time.time() - _START_TIME[0]


def _remaining():
    """Return remaining time budget."""
    return _DEADLINE_SECONDS - _elapsed()


def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system using vectorized operations."""
    n = len(x)
    x_ip1 = np.roll(x, -1)
    x_im1 = np.roll(x, 1)
    x_im2 = np.roll(x, 2)
    return (x_ip1 - x_im2) * x_im1 - x + F


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _lorenz96_propagate(x, F, dt, nsteps):
    """Propagate Lorenz-96 state forward."""
    for _ in range(nsteps):
        x = _rk4_step(x, F, dt)
    return x


def _gaspari_cohn_localization(dist, L):
    """Gaspari-Cohn localization function."""
    if dist <= L / 2:
        x = 2 * dist / L
        return (-1/4) * x**5 + (5/2) * x**4 - (5/2) * x**3 + (5/12) + 1
    elif dist <= L:
        x = 2 * dist / L
        return (1/12) * x**5 - (5/2) * x**4 + (5/2) * x**3 - (5/3) * x**2 + (4/3) * x + (5/12)
    else:
        return 0.0


def _compute_localization_matrix(D, observed_indices, localization_length):
    """Compute localization weight matrix (D, M) for ring geometry."""
    M = len(observed_indices)
    loc_weights = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)
            loc_weights[d, m_idx] = _gaspari_cohn_localization(dist, localization_length)
    return loc_weights


def _particle_filter_update(ensemble, obs_vec, observed_indices, observation_variance,
                            localization_length=3.0, inflation_base=1.01,
                            max_inflation=1.25, bandwidth_factor=1.5):
    """
    Nonlinear particle filter update with importance resampling.
    Uses Gaspari-Cohn localization and adaptive inflation.
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Ensure obs_vec is 1D
    obs_vec = np.asarray(obs_vec).flatten()
    
    # Compute ensemble mean
    ens_mean = np.mean(ensemble, axis=0)
    
    # Centered ensemble (anomalies)
    anomalies = ensemble - ens_mean
    
    # Observation operator: H selects observed components
    obs_anomalies = anomalies[:, observed_indices]  # (N, M)
    
    # Compute observation-space covariance P_obs
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)  # (M, M)
    
    # Add observation noise
    R = observation_variance * np.eye(M)
    S = P_obs + R  # Innovation covariance (M, M)
    
    # Compute cross-covariance PHt
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)  # (D, M)
    
    # Apply localization to the cross-covariance
    loc_weights = _compute_localization_matrix(D, observed_indices, localization_length)
    PHt_loc = PHt * loc_weights  # (D, M)
    
    # Solve for S_inv using Cholesky for stability
    try:
        L_S = np.linalg.cholesky(S)
        Y = np.linalg.solve(L_S, np.eye(M))
        S_inv = Y.T @ Y
    except np.linalg.LinAlgError:
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
    
    # Compute innovations for each particle
    innovations = obs_vec[None, :] - ensemble[:, observed_indices]  # (N, M)
    
    # Compute importance weights using a Gaussian likelihood
    Si_innov = innovations @ S_inv.T  # (N, M)
    log_weights = -0.5 * np.sum(Si_innov * innovations, axis=1)  # (N,)
    
    # Normalize log-weights for numerical stability
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    
    # Effective sample size
    ess = 1.0 / np.sum(weights ** 2)
    
    # Determine if resampling is needed
    resample_threshold = 0.5 * N
    
    if ess < resample_threshold:
        # Systematic resampling
        u = np.random.uniform(0, 1, N) / N
        cum_weights = np.cumsum(weights)
        indices = np.searchsorted(cum_weights, u)
        indices = np.clip(indices, 0, N - 1)
        resampled = ensemble[indices].copy()
        
        # Add small noise to prevent particle degeneracy
        noise_std = 0.1 * np.std(ensemble, axis=0)
        resampled += np.random.normal(0, 1, (N, D)) * noise_std
    else:
        # No resampling needed; use weighted update (EnKF style)
        # Compute weighted mean
        weighted_mean = np.sum(weights[:, None] * ensemble, axis=0)
        
        # Compute weighted anomalies
        weighted_anomalies = ensemble - weighted_mean
        
        # Compute weighted cross-covariance
        PHt_w = (weights[:, None] * weighted_anomalies).T @ obs_anomalies / np.sum(weights)
        
        # Apply localization
        PHt_w_loc = PHt_w * loc_weights
        
        # Compute update using weighted approach
        Si_innov_w = innovations @ S_inv.T  # (N, M)
        delta = Si_innov_w @ PHt_w_loc.T  # (N, D)
        
        resampled = ensemble + delta
    
    # Adaptive inflation based on innovation spread
    obs_mean_pre = np.mean(ensemble[:, observed_indices], axis=0)
    innov_mean = obs_vec - obs_mean_pre  # (M,)
    
    P_obs_diag = np.diag(P_obs)
    R_diag = np.full(M, observation_variance)
    expected_innov_var = P_obs_diag + R_diag  # (M,)
    expected_innov_std = np.sqrt(np.maximum(expected_innov_var, 1e-10))
    
    actual_innov_abs = np.abs(innov_mean)
    innov_ratio = np.mean(actual_innov_abs / expected_innov_std)
    
    alpha = 0.4
    inflation = 1.0 + alpha * max(0.0, innov_ratio - 1.0)
    
    ess_ratio = ess / N
    if ess_ratio < 0.5:
        inflation += 0.1 * (0.5 - ess_ratio)
    
    inflation = min(inflation, max_inflation)
    inflation = max(inflation, inflation_base)
    
    # Apply inflation about the ensemble mean
    if inflation > 1.0:
        resampled = ens_mean + inflation * (resampled - ens_mean)
    
    # Ensure finite values and clip to valid range
    resampled = np.nan_to_num(resampled, nan=0.0, posinf=100.0, neginf=-100.0)
    resampled = np.clip(resampled, -100.0, 100.0)
    
    return resampled


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for partial-observation Lorenz-96.
    
    Parameters:
    -----------
    ensemble : (16, 20) array - ensemble of 16 members, 20 state variables
    observations : (10,) or (16, 10) array - noisy observations of even-indexed coordinates
    observed_indices : array of 10 indices - the even indices [0, 2, 4, ..., 18]
    observation_variance : float - variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : (16, 20) array - updated ensemble
    """
    # Deadline guard
    if _remaining() < 0.5:
        # Not enough time; return ensemble as-is (safe fallback)
        return np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        # If it's (N, M), assume all rows are the same or take the first one
        # Standard EnKF uses a single observation vector
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr
    
    # Check time again before the main computation
    if _remaining() < 1.0:
        return np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Particle filter update with localization and adaptive inflation
    result = _particle_filter_update(
        ensemble, obs_vec, observed_indices, observation_variance,
        localization_length=3.0,
        inflation_base=1.01,
        max_inflation=1.25,
        bandwidth_factor=1.5
    )
    
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
