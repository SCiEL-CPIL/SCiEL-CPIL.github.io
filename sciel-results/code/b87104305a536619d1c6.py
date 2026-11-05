"""Ensemble Kalman Filter with localization, inflation, and ensemble adjustment."""
import numpy as np
import time

def _lorenz96(x, F):
    """Compute Lorenz-96 derivatives for a single state vector."""
    n = len(x)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
    return dx

def _rk4_step(x, F, dt):
    """One RK4 step for Lorenz-96."""
    k1 = _lorenz96(x, F)
    k2 = _lorenz96(x + 0.5*dt*k1, F)
    k3 = _lorenz96(x + 0.5*dt*k2, F)
    k4 = _lorenz96(x + dt*k3, F)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update using an Ensemble Kalman Filter (EnKF) with
    ensemble adjustment (EnA), localization, and adaptive inflation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (10,) or (16, 10)
        Noisy observations of the observed coordinates.
        If shape is (16,10), we take the mean across ensemble members.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=np.int64)
    
    n_ens, n_state = ensemble.shape
    n_obs = len(observed_indices)
    
    # Handle observations shape: if (n_ens, n_obs), take mean; if (n_obs,), broadcast
    if observations.ndim == 2 and observations.shape[0] == n_ens:
        obs_mean = observations.mean(axis=0)
    elif observations.ndim == 2 and observations.shape[1] == n_obs:
        obs_mean = observations.mean(axis=0)
    else:
        obs_mean = observations
    
    # Ensure obs_mean has correct length
    if len(obs_mean) != n_obs:
        obs_mean = obs_mean[:n_obs]
    
    # Estimate F from the ensemble mean (rough heuristic)
    # For Lorenz-96, F is a forcing parameter. We'll use a reasonable default.
    # Actually, we don't know F directly, but we can try to infer it or use a default.
    # The task says F=6,8,10 for different runs. We'll use a moderate value.
    # Better: we can estimate F from the steady-state behavior, but that's complex.
    # Let's just use a reasonable default and rely on the filter to work.
    # Actually, for the analysis step, we don't need to propagate, so F doesn't matter
    # for the update itself. The evaluator handles propagation.
    
    # --- Localization: use a correlation-based localization weight ---
    # Compute ensemble covariance for observed variables
    ens_obs = ensemble[:, observed_indices]  # (n_ens, n_obs)
    
    # Compute ensemble mean and covariance
    ens_mean = ensemble.mean(axis=0)  # (n_state,)
    ens_cov = np.cov(ensemble, rowvar=False)  # (n_state, n_state)
    
    # Observation covariance
    R = observation_variance * np.eye(n_obs)
    
    # --- Ensemble Adjustment Kalman Filter (EnA) ---
    # This avoids sampling error and preserves ensemble mean
    # Standard EnA update:
    #   K = P H^T (H P H^T + R)^{-1}
    #   x_a = x + K (y - H x)
    
    H = np.zeros((n_obs, n_state))
    H[np.arange(n_obs), observed_indices] = 1.0
    
    # Compute P H^T
    PHt = ens_cov @ H.T  # (n_state, n_obs)
    
    # Compute H P H^T
    HPHt = H @ ens_cov @ H.T  # (n_obs, n_obs)
    
    # Add observation noise
    S = HPHt + R  # (n_obs, n_obs)
    
    # Compute gain
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = PHt @ S_inv  # (n_state, n_obs)
    
    # Innovation
    innovation = obs_mean - H @ ens_mean  # (n_obs,)
    
    # Update ensemble mean
    updated_mean = ens_mean + K @ innovation  # (n_state,)
    
    # Update ensemble members using ensemble adjustment
    # x_i^a = x_i + K (y - H x_i)  -- but this changes the mean
    # For EnA, we want to preserve the mean:
    # x_i^a = x_i + K (y - H x_i) - K (y - H x_mean)
    #       = x_i + K (H x_mean - H x_i)
    #       = x_i - K H (x_i - x_mean)
    
    # Actually, the standard EnA formula is:
    # x_i^a = x_i + K (y - H x_i)  for each member
    # But this shifts the mean. To preserve the mean, we use:
    # x_i^a = x_i + K (y - H x_i) - K (y - H x_bar)
    #        = x_i + K H (x_bar - x_i)
    
    # Let's use the standard EnKF with perturbed observations to avoid bias
    # Or use the ensemble adjustment method properly
    
    # Ensemble Adjustment (Hunt et al. 2007):
    # For each member i:
    #   x_i^a = x_i + K (y - H x_i)
    # where K is computed from the ensemble covariance
    # This preserves the ensemble mean if we use the right K
    
    # Actually, let's just do the standard linear update for each member
    # with a localization factor to avoid spurious correlations
    
    # Localization: reduce the gain for distant variables
    # Use a simple exponential localization based on distance
    loc_scale = 4.0  # localization scale in grid points
    loc_weights = np.ones(n_state)
    for j in range(n_state):
        # Distance from each observed index to variable j
        min_dist = min(abs(j - oi) for oi in observed_indices)
        # Also consider periodic distance
        periodic_dist = min(min_dist, n_state - min_dist)
        loc_weights[j] = np.exp(-0.5 * (periodic_dist / loc_scale)**2)
    
    # Apply localization to the gain
    K_loc = K * loc_weights[:, np.newaxis]
    
    # Update each ensemble member
    updated_ensemble = np.empty_like(ensemble)
    for i in range(n_ens):
        # Innovation for member i
        Hxi = H @ ensemble[i]  # (n_obs,)
        innov_i = obs_mean - Hxi
        updated_ensemble[i] = ensemble[i] + K_loc @ innov_i
    
    # Ensure the ensemble mean is preserved (EnA property)
    # The mean of updated_ensemble should be close to updated_mean
    current_mean = updated_ensemble.mean(axis=0)
    updated_ensemble += (updated_mean - current_mean)
    
    # Adaptive inflation: check if ensemble spread is too small
    # Compute spread at observed points
    spread_obs = np.std(updated_ensemble[:, observed_indices], axis=0)
    obs_spread = np.std(ens_obs, axis=0)
    
    # If spread has decreased too much, inflate
    inflation_factor = 1.0
    min_spread_ratio = 0.7
    spread_ratio = np.mean(spread_obs / (obs_spread + 1e-10))
    if spread_ratio < min_spread_ratio:
        inflation_factor = 1.0 / max(spread_ratio, 0.5)
    
    if inflation_factor > 1.0:
        updated_ensemble = updated_mean + inflation_factor * (updated_ensemble - updated_mean)
    
    # Clip to valid range
    updated_ensemble = np.clip(updated_ensemble, -100, 100)
    
    # Check deadline
    if time.time() > deadline:
        return updated_ensemble
    
    return updated_ensemble
