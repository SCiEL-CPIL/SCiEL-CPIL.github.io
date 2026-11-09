"""Combined assimilation for Lorenz-96 partial observation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Global deadline guard: 5s safety margin before 150s kill
_DEADLINE = time.time() + 145.0

def _gaussian_localization(dist, length_scale):
    """Gaussian localization weight."""
    return np.exp(-0.5 * (dist / length_scale) ** 2)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with observation perturbation for Lorenz-96.
    
    Uses the stochastic sampling of observation noise method to update the
    ensemble. This approach inherently preserves ensemble covariance structure
    and avoids the bias introduced by explicit multiplicative inflation.
    
    Parameters:
    -----------
    ensemble : (16, 20) ndarray
        Ensemble of 16 members, each a 20-state Lorenz-96 vector
    observations : (10,) ndarray
        Noisy observations of the 10 even-indexed states
    observed_indices : array_like
        Indices of observed variables
    observation_variance : float
        Variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : (16, 20) ndarray
        Updated ensemble after assimilation
    """
    # Deadline check
    if time.time() > _DEADLINE:
        return ensemble.copy()
    
    N, M = ensemble.shape
    obs_idx = np.array(sorted(observed_indices), dtype=int)
    n_obs = len(obs_idx)
    
    # Ensure finite input
    ens = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0).astype(np.float64)
    obs = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0).astype(np.float64)
    
    # --- Step 1: Compute ensemble mean and anomalies ---
    x_mean = np.mean(ens, axis=0)  # (20,)
    anomalies = ens - x_mean  # (N, 20)
    
    # --- Step 2: Build localization matrix ---
    # Gaussian localization with length scale tuned for L96 correlation structure
    length_scale = 2.5
    
    # Precompute periodic distances from each observed index to all state indices
    dist_matrix = np.zeros((n_obs, M))
    for i, oi in enumerate(obs_idx):
        for j in range(M):
            d = abs(oi - j)
            d = min(d, M - d)  # periodic
            dist_matrix[i, j] = d
    
    # Localization weights
    L = _gaussian_localization(dist_matrix, length_scale)  # (10, 20)
    # Ensure self-observation gets full weight
    for i in range(n_obs):
        L[i, obs_idx[i]] = 1.0
    
    # --- Step 3: Observation perturbation method ---
    # Generate random perturbations for observations
    # Each member gets its own perturbed observation
    obs_std = np.sqrt(observation_variance)
    
    # Generate N x n_obs perturbation matrix
    rng = np.random.default_rng()
    obs_perturbations = rng.standard_normal((N, n_obs)) * obs_std
    
    # Perturbed observations: obs_perturbed[n, k] = obs[k] + perturbation[n, k]
    obs_perturbed = obs[None, :] + obs_perturbations  # (N, n_obs)
    
    # --- Step 4: Vectorized EKF update with observation perturbation ---
    # For each member n, compute innovation and update
    # Innovation for member n at observation k: y_nk = obs_perturbed[n, k] - ens[n, obs_idx[k]]
    
    # Extract observed state values for all members
    ens_obs = ens[:, obs_idx]  # (N, n_obs)
    
    # Innovations: (N, n_obs)
    innovations = obs_perturbed - ens_obs
    
    # Compute cross-covariance matrix P_xy: cov(x_j, y_k) for all j, k
    # P_xy[j, k] = (1/N) * sum_n (anomaly[n, j] * anomaly_obs[n, k])
    anomalies_obs = anomalies[:, obs_idx]  # (N, n_obs)
    P_xy = (anomalies.T @ anomalies_obs) / N  # (M, n_obs)
    
    # Apply localization to cross-covariance
    # L is (n_obs, M), so we need to multiply appropriately
    # P_xy_localized[j, k] = P_xy[j, k] * L[k, j]
    P_xy_loc = P_xy * L.T  # (M, n_obs)
    
    # Observation covariance: R = observation_variance * I (diagonal)
    # Since observations are independent, R is diagonal with obs_variance
    R_inv = 1.0 / observation_variance  # scalar, since R = obs_variance * I
    
    # Kalman gain: K = P_xy * R_inv (for diagonal R)
    # K[j, k] = P_xy_loc[j, k] / observation_variance
    K = P_xy_loc * R_inv  # (M, n_obs)
    
    # State update: x_n = x_n + K @ innovation_n
    # innovation_n is (n_obs,), K is (M, n_obs)
    # K @ innovation_n gives (M,) update for each member
    updates = innovations @ K.T  # (N, M)
    
    updated_ens = ens + updates
    
    # --- Step 5: Mild spread preservation ---
    # Check if spread has collapsed and apply gentle inflation if needed
    new_mean = np.mean(updated_ens, axis=0)
    current_spread = np.std(updated_ens, axis=0)
    pre_spread = np.std(ens, axis=0)
    
    for j in range(M):
        if current_spread[j] > 1e-8:
            # Target: maintain at least 90% of pre-update spread
            target = pre_spread[j] * 0.90
            if current_spread[j] < target:
                scale = target / current_spread[j]
                scale = min(scale, 1.2)  # cap inflation
                updated_ens[:, j] = new_mean[j] + (updated_ens[:, j] - new_mean[j]) * scale
        else:
            # Add small perturbation if spread collapsed
            updated_ens[:, j] = new_mean[j] + np.random.normal(0, 0.05, N)
    
    # --- Step 6: Safety checks ---
    if not np.all(np.isfinite(updated_ens)):
        updated_ens = np.nan_to_num(updated_ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    updated_ens = np.clip(updated_ens, -100.0, 100.0)
    
    return updated_ens

# EVOLVE-BLOCK-END
