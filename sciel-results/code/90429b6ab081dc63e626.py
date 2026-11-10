"""Lorenz-96 ensemble assimilation with stochastic EKPF and localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START
_N = 20
_WINDOW_HALF = 3

def _gaspari_cohn(d, c):
    """Gaspari-Cohn C2 correlation function."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    m1 = d <= 0.5 * c
    m2 = (d > 0.5 * c) & (d <= 1.5 * c)
    m3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    dm = d[m1]
    result[m1] = (-5/4)*(dm/c)**5 + 5*(dm/c)**4 - 10*(dm/c)**3 + 5*dm
    
    dm = d[m2]
    result[m2] = (1/12)*(dm/c)**5 - (5/4)*(dm/c)**4 + (5/2)*(dm/c)**3 - 5*(dm/c)**2 + (25/12)*(dm/c) + 1/2
    
    dm = d[m3]
    result[m3] = -(1/16)*(dm/c)**5 + (1/4)*(dm/c)**4 - (15/8)*(dm/c)**3 + (5/2)*(dm/c)**2 - (25/12)*(dm/c) + 105/48
    
    return result

def _periodic_distance(i, j, N):
    """Minimum distance on a ring of size N."""
    d = abs(i - j)
    return min(d, N - d)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Particle Filter (EKPF) for Lorenz-96.
    
    Two-stage update:
    1. Deterministic local EnKF pass for the mean adjustment
    2. Stochastic particle resampling with correlated noise to maintain spread
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations
    observed_indices: array of observed state indices
    observation_variance: scalar observation noise variance
    
    Returns:
    Updated ensemble (M, N)
    """
    t0 = time.time()
    M, N = ensemble.shape
    K = len(observed_indices)
    result = ensemble.astype(float).copy()
    
    # Build observed index set for fast lookup
    obs_set = set(int(x) for x in observed_indices)
    obs_map = {int(observed_indices[k]): k for k in range(K)}
    
    # Adaptive multiplicative inflation (moderate)
    if M > 2:
        spread = np.std(ensemble[:, list(obs_set)], axis=0)
        obs_err = np.sqrt(observation_variance)
        ratio = np.mean(spread) / obs_err if obs_err > 0 else 1.0
        if ratio < 1.0:
            inflation = 1.0 + 0.03 * (1.0 - ratio)
        else:
            inflation = 1.003
        inflation = float(np.clip(inflation, 1.003, 1.03))
    else:
        inflation = 1.003
    
    mean = np.mean(result, axis=0)
    result = mean + inflation * (result - mean)
    
    # Correlation length for localization
    corr_len = 4.0
    
    # Precompute localization weights for each center i
    loc_weights_by_i = []
    for i in range(N):
        dists = np.array([_periodic_distance(i, (i + j) % N, N) for j in range(-3, 4)])
        w = _gaspari_cohn(dists, corr_len)
        w[3] = 1.0
        loc_weights_by_i.append(w)
    
    # Time budget: 140s max (safety margin from 150s)
    time_budget = 140.0
    
    # STAGE 1: Deterministic local EnKF update
    for i in range(N):
        if time.time() - t0 > time_budget:
            break
        
        local_idx = [(i + j) % N for j in range(-3, 4)]
        
        local_obs_pos = []
        for l, g in enumerate(local_idx):
            if g in obs_set:
                local_obs_pos.append(l)
        
        if len(local_obs_pos) == 0:
            continue
        
        n_obs_local = len(local_obs_pos)
        
        local_ens = result[:, local_idx]
        
        y_local = np.array([observations[obs_map[local_idx[l]]] for l in local_obs_pos])
        
        local_mean = np.mean(local_ens, axis=0)
        local_anoms = local_ens - local_mean
        
        w = loc_weights_by_i[i]
        local_anoms_loc = local_anoms * w[np.newaxis, :]
        
        C = (local_anoms_loc.T @ local_anoms_loc) / (M - 1)
        
        H = np.zeros((n_obs_local, 7))
        for k, l in enumerate(local_obs_pos):
            H[k, l] = 1.0
        
        HC = H @ C
        S = HC @ H.T + observation_variance * np.eye(n_obs_local)
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        K_gain = C @ H.T @ S_inv
        
        for m in range(M):
            Hx_m = H @ local_ens[m, :]
            innov_m = y_local - Hx_m
            local_ens_new = local_ens[m, :] + K_gain @ innov_m
            result[m, local_idx] = local_ens_new
    
    # STAGE 2: Stochastic particle resampling (EKPF step)
    # This is the key structural change: treat ensemble members as particles
    # and resample from a Gaussian proposal centered at the post-EnKF mean
    # with noise scaled to maintain proper spread
    
    if time.time() - t0 > time_budget - 10.0:
        # Not enough time for stochastic step, return deterministic result
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        result = np.clip(result, -100.0, 100.0)
        return result
    
    # Compute post-EnKF ensemble statistics
    post_mean = np.mean(result, axis=0)
    post_anoms = result - post_mean  # (M, N)
    
    # Estimate analysis error covariance (diagonal approximation for efficiency)
    # Use localized variance to avoid spurious long-range correlations
    diag_var = np.zeros(N)
    for i in range(N):
        # Local window variance with localization
        local_idx = [(i + j) % N for j in range(-3, 4)]
        w = loc_weights_by_i[i]
        local_anoms = post_anoms[:, local_idx]
        # Weighted variance
        diag_var[i] = np.mean(np.sum((local_anoms * w[np.newaxis, :])**2, axis=1))
    
    # Target spread: balance between analysis spread and observation error
    # For observed variables, target ~ sqrt(obs_var) 
    # For unobserved, target ~ sqrt(diag_var)
    target_spread = np.sqrt(diag_var)
    # For observed indices, pull toward observation error scale
    for idx in obs_set:
        target_spread[idx] = np.sqrt(observation_variance) * 0.8
    
    # Ensure minimum spread to prevent collapse
    min_spread = 0.3
    target_spread = np.maximum(target_spread, min_spread)
    
    # Generate correlated stochastic perturbations
    # Use a simple approach: perturb each variable with independent noise
    # scaled by target spread, but correlate with neighboring variables
    # via a local averaging kernel
    
    rng = np.random.default_rng()
    
    # Generate base noise
    noise = rng.standard_normal((M, N))
    
    # Apply local smoothing to create spatially correlated noise
    # This mimics the physical correlations in Lorenz-96
    smoothed_noise = np.zeros_like(noise)
    for i in range(N):
        local_idx = [(i + j) % N for j in range(-2, 3)]
        w = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        smoothed_noise[:, i] = np.mean(noise[:, local_idx] * w[np.newaxis, :], axis=1)
    
    # Normalize to unit variance
    noise_var = np.var(smoothed_noise, axis=0)
    noise_var[noise_var < 1e-10] = 1.0
    smoothed_noise = smoothed_noise / np.sqrt(noise_var)[np.newaxis, :]
    
    # Scale by target spread
    stochastic_pert = smoothed_noise * target_spread[np.newaxis, :]
    
    # Blend: keep some of the EnKF structure, add stochastic component
    # alpha controls how much stochastic noise we add
    # Too much noise degrades the mean; too little doesn't help spread
    alpha = 0.15
    
    # New ensemble: post-EnKF + stochastic perturbation
    # Center at post_mean to preserve the mean
    new_result = post_mean[np.newaxis, :] + (result - post_mean[np.newaxis, :]) * (1 - alpha) + stochastic_pert * alpha
    
    # Ensure the ensemble mean is preserved (remove any bias from noise)
    new_mean = np.mean(new_result, axis=0)
    new_result = new_result - new_mean[np.newaxis, :] + post_mean[np.newaxis, :]
    
    result = new_result
    
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
# EVOLVE-BLOCK-END
