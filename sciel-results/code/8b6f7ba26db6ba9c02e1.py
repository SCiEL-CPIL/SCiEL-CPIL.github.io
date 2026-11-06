"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_start_time = None
_prev_ens_mean = None
_prev_obs = None

def _get_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _lorenz96_mean_field(x, F):
    """Compute mean field contribution for localization."""
    N = len(x)
    mf = np.zeros(N)
    for i in range(N):
        # Estimate local mean from neighbors
        neighbors = [x[(i-1)%N], x[(i+1)%N], x[(i-2)%N], x[(i+2)%N]]
        mf[i] = np.mean(neighbors)
    return mf

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Advanced Ensemble Kalman Filter with:
    - Gaussian localization with adaptive length scale
    - Innovation covariance with localization
    - Spread-based adaptive inflation
    - Temporal consistency via previous state
    - Deadline guard
    """
    global _prev_ens_mean, _prev_obs
    
    M, N = ensemble.shape
    obs_var = observation_variance
    obs_idx = np.asarray(observed_indices, dtype=int)
    n_obs = len(obs_idx)
    
    # Deadline guard
    elapsed = _get_time()
    if elapsed > 140.0:
        result = np.array(ensemble, dtype=np.float64).copy()
        obs_arr = np.asarray(observations, dtype=np.float64)
        if obs_arr.ndim == 1:
            obs_arr = np.tile(obs_arr, (M, 1))
        result[:, obs_idx] += 0.5 * (obs_arr - result[:, obs_idx])
        np.clip(result, -100.0, 100.0, out=result)
        return result
    
    # Copy ensemble
    result = np.array(ensemble, dtype=np.float64).copy()
    
    # --- Compute ensemble statistics ---
    ens_mean = np.mean(result, axis=0)
    anomalies = result - ens_mean  # (M, N)
    
    # --- Observation processing ---
    obs_arr = np.asarray(observations, dtype=np.float64)
    if obs_arr.ndim == 1:
        obs_arr = np.tile(obs_arr, (M, 1))
    
    H_ens = result[:, obs_idx]  # (M, n_obs)
    innovation = obs_arr - H_ens  # (M, n_obs)
    
    # Anomalies for observed variables
    anom_obs = anomalies[:, obs_idx]  # (M, n_obs)
    
    # --- Compute cross-covariance ---
    cross_cov = (anomalies.T @ anom_obs) / (M - 1)  # (N, n_obs)
    
    # --- Gaussian localization ---
    # Adaptive length scale based on observation density
    obs_spacing = N / n_obs if n_obs > 0 else N
    L_loc = max(2.0, min(4.0, 0.6 * obs_spacing))
    
    # Pre-compute localization weights (periodic distance)
    loc_weights = np.zeros((N, n_obs))
    for j_idx, j in enumerate(obs_idx):
        for i in range(N):
            d = abs(i - j)
            d = min(d, N - d)  # periodic distance
            loc_weights[i, j_idx] = np.exp(-0.5 * (d / L_loc) ** 2)
    
    # Apply localization to cross-covariance
    cross_cov_loc = cross_cov * loc_weights
    
    # --- Innovation covariance with localization ---
    # Compute localized innovation covariance
    innov_cov = (anom_obs.T @ anom_obs) / (M - 1)
    
    # Add localization to innovation covariance diagonal
    # This prevents spurious correlations
    for j_idx, j in enumerate(obs_idx):
        # Count effective neighbors within localization range
        eff_neighbors = 0
        for i in range(N):
            d = abs(i - j)
            d = min(d, N - d)
            if d <= L_loc:
                eff_neighbors += 1
        # Reduce off-diagonal contributions by localization
        if eff_neighbors < N:
            loc_factor = max(0.5, min(1.0, 1.0 - 0.1 * (N - eff_neighbors) / N))
            for k in range(n_obs):
                if k != j_idx:
                    d = abs(obs_idx[j_idx] - obs_idx[k])
                    d = min(d, N - d)
                    w = np.exp(-0.5 * (d / L_loc) ** 2)
                    innov_cov[j_idx, k] *= w
                    innov_cov[k, j_idx] *= w
    
    innov_cov += obs_var * np.eye(n_obs)
    
    # Regularize for numerical stability
    innov_cov += 1e-6 * np.eye(n_obs)
    
    # Invert innovation covariance
    try:
        innov_cov_inv = np.linalg.inv(innov_cov)
    except np.linalg.LinAlgError:
        innov_cov_inv = np.linalg.pinv(innov_cov)
    
    # --- Kalman gain ---
    K = cross_cov_loc @ innov_cov_inv  # (N, n_obs)
    
    # Apply update
    update = (K @ innovation.T).T  # (M, N)
    result = result + update
    
    # --- Temporal consistency: blend with previous analysis ---
    if _prev_ens_mean is not None and _prev_obs is not None:
        # Weight based on how well previous analysis matched current observations
        prev_pred = _prev_ens_mean[obs_idx]
        prev_innov = _prev_obs - prev_pred
        prev_innov_norm = np.mean(prev_innov ** 2)
        
        # If previous analysis was good, trust it more
        if prev_innov_norm < 2.0 * obs_var:
            blend_weight = 0.1  # 10% weight to previous
            new_mean = np.mean(result, axis=0)
            result = (1 - blend_weight) * result + blend_weight * (_prev_ens_mean[None, :] + (result - new_mean))
    _prev_ens_mean = np.mean(result, axis=0)
    _prev_obs = obs_arr[0] if obs_arr.shape[0] > 1 else obs_arr
    
    # --- Adaptive multiplicative inflation ---
    # Compute ensemble spread for observed variables
    ens_spread_obs = np.std(result[:, obs_idx], axis=0, ddof=1)
    
    # Expected spread from observation noise
    expected_spread = np.sqrt(obs_var)
    
    # Spread-skill ratio based inflation
    min_spread = 0.5 * expected_spread
    max_inflation = 1.3
    
    inflation_factors = np.ones(n_obs)
    for j in range(n_obs):
        if ens_spread_obs[j] < min_spread:
            inflation_factors[j] = min_spread / max(ens_spread_obs[j], 1e-10)
            inflation_factors[j] = min(inflation_factors[j], max_inflation)
    
    # Apply inflation to all variables (global)
    avg_inflation = np.mean(inflation_factors)
    if avg_inflation > 1.001:
        new_mean = np.mean(result, axis=0)
        result = new_mean + avg_inflation * (result - new_mean)
    
    # --- Final safety checks ---
    np.clip(result, -100.0, 100.0, out=result)
    
    if not np.all(np.isfinite(result)):
        result = np.array(ensemble, dtype=np.float64).copy()
        result[:, obs_idx] += 0.5 * (obs_arr - result[:, obs_idx])
        np.clip(result, -100.0, 100.0, out=result)
    
    return result

# EVOLVE-BLOCK-END
