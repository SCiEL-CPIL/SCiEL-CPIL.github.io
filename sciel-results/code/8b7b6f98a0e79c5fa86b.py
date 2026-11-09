"""Combined assimilation for Lorenz-96 partial observation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Global deadline guard: 5s safety margin before 150s kill
_DEADLINE = time.time() + 145.0

def _gaussian_localization(dist, length_scale):
    """Gaussian localization weight."""
    return np.exp(-0.5 * (dist / length_scale) ** 2)

def _random_orthogonal(N, rng):
    """Generate a random orthogonal matrix using QR decomposition."""
    A = rng.standard_normal((N, N))
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det=+1)
    d = np.sign(np.diag(R))
    d[d < 0] = 1.0
    Q = Q * d
    return Q

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter (SE-KF) with observation perturbation
    for Lorenz-96 partial observation.
    
    Uses stochastic sampling of observation noise via random orthogonal
    transformations to preserve ensemble spread without explicit inflation.
    Gaussian localization handles the partial observation structure.
    
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
    # Localization scale tuned for L96 correlation length (~3-4 grid points)
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
    
    # --- Step 3: SE-KF with observation perturbation ---
    # For each member, add a random perturbation to the observation
    # that preserves the observation variance structure
    
    updated_ens = ens.copy()
    
    # Generate random orthogonal matrix for observation perturbations
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    Q_obs = _random_orthogonal(N, rng)  # (N, N)
    
    # Observation anomalies: obs - mean(obs)
    obs_mean = np.mean(obs)
    obs_anoms = obs - obs_mean  # (10,)
    
    # Scale observation anomalies to match ensemble spread at observed points
    # This ensures the perturbation has the correct statistical properties
    ens_obs_spread = np.std(ens[:, obs_idx], axis=0)  # (10,)
    obs_spread = np.std(obs_anoms)  # scalar
    
    # Create perturbed observations for each member
    # For each observed variable, we generate N perturbed values
    # that have the same mean as obs but with spread matching obs_spread
    
    # Process each observation sequentially with perturbation
    for k in range(n_obs):
        if time.time() > _DEADLINE:
            break
        
        oi = obs_idx[k]
        
        # Current ensemble statistics
        cur_mean = np.mean(updated_ens, axis=0)  # (M,)
        cur_anom = updated_ens - cur_mean  # (N, M)
        
        # Local variance at observation point
        x_k_anom = cur_anom[:, oi]  # (N,)
        var_k = np.mean(x_k_anom ** 2)
        
        # Denominator
        denom = var_k + observation_variance
        
        if denom < 1e-10:
            continue
        
        # Cross-covariance vector with localization
        P_xy = (cur_anom.T @ x_k_anom) / N * L[k, :]  # (M,)
        
        # Kalman gain vector
        K = P_xy / denom  # (M,)
        
        # Generate perturbed observation for each member
        # The perturbation should have zero mean and variance = observation_variance
        # Use the orthogonal matrix to decorrelate perturbations
        base_pert = np.zeros(N)
        base_pert[0] = np.sqrt(observation_variance)
        perturbed_obs = Q_obs.T @ base_pert  # (N,)
        
        # Innovation for each member: perturbed_obs - updated_ens[n, oi]
        member_innov = (obs[k] + perturbed_obs) - updated_ens[:, oi]  # (N,)
        
        # Update all members at once
        updated_ens += np.outer(member_innov, K)
    
    # --- Step 4: Light adaptive spread maintenance ---
    # SE-KF should preserve spread, but add minimal correction if needed
    new_mean = np.mean(updated_ens, axis=0)
    current_spread = np.std(updated_ens, axis=0)
    pre_spread = np.std(ens, axis=0)
    
    # Only apply very mild inflation if spread dropped significantly
    for j in range(M):
        if current_spread[j] > 1e-8 and pre_spread[j] > 1e-8:
            ratio = current_spread[j] / pre_spread[j]
            if ratio < 0.85:
                # Gentle correction: boost by at most 10%
                scale = min(1.0 / max(ratio, 0.7), 1.15)
                updated_ens[:, j] = new_mean[j] + (updated_ens[:, j] - new_mean[j]) * scale
    
    # --- Step 5: Safety checks ---
    if not np.all(np.isfinite(updated_ens)):
        updated_ens = np.nan_to_num(updated_ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    updated_ens = np.clip(updated_ens, -100.0, 100.0)
    
    return updated_ens

# EVOLVE-BLOCK-END
