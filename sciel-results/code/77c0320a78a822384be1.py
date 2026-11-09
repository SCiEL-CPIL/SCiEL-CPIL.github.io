"""Lorenz-96 ensemble assimilation with Stochastic EnKF and adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter analysis update for Lorenz-96.
    Uses adaptive multiplicative inflation, Gaussian localization, and stochastic perturbations.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # --- Step 1: Adaptive Multiplicative Inflation ---
    ens_mean = np.mean(ens, axis=0)
    ens_centered = ens - ens_mean
    
    # Compute spread at observed locations
    ens_var_obs = np.var(ens[:, obs_idx], axis=0)
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-15))
    
    # Target spread: slightly larger than observation std
    target_spread = np.sqrt(R_val) * 1.3
    
    # Inflation factor using tanh-based smooth clamp
    ratio = target_spread / np.maximum(current_spread, 1e-15)
    mean_ratio = np.mean(ratio)
    
    # Use tanh to smoothly map the ratio to a reasonable inflation range
    # When ratio is 1.0, inflation should be close to 1.0
    # When ratio is large, inflation should approach ~1.15
    # tanh(x) maps (-inf, inf) to (-1, 1)
    # We want: inflation = 1.0 + 0.15 * tanh(0.5 * (mean_ratio - 1.0))
    # At mean_ratio=1.0: inflation = 1.0 + 0.15*tanh(0) = 1.0
    # At mean_ratio=1.5: inflation = 1.0 + 0.15*tanh(0.25) ≈ 1.037
    # At mean_ratio=2.0: inflation = 1.0 + 0.15*tanh(0.5) ≈ 1.071
    # At mean_ratio=3.0: inflation = 1.0 + 0.15*tanh(1.0) ≈ 1.114
    
    raw_inflation = 1.0 + 0.15 * np.tanh(0.5 * (mean_ratio - 1.0))
    inflation = np.clip(raw_inflation, 1.0, 1.18)
    
    # Apply multiplicative inflation
    ens = ens_mean + inflation * ens_centered
    
    # --- Step 2: Localization Weights ---
    indices = np.arange(n_vars)
    # Circular distance for periodic boundary
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    
    # Gaussian localization with sigma=3.0 for L96 correlation length
    sigma = 3.0
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)
    
    # --- Step 3: Compute Covariances ---
    ens_mean_new = np.mean(ens, axis=0)
    A = ens - ens_mean_new  # anomalies (n_members, n_vars)
    
    A_obs = ens[:, obs_idx] - ens_mean_new[obs_idx]  # (n_members, n_obs)
    
    # Innovation
    innovation = obs - ens_mean_new[obs_idx]  # (n_obs,)
    
    # Observation-space covariance
    P_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # (n_obs, n_obs)
    S = P_oo + R_val * np.eye(n_obs)
    
    # Enforce symmetry
    S = 0.5 * (S + S.T)
    
    # Robust Cholesky Decomposition
    L = None
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # Try with increasing jitter
        for jitter in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
            try:
                L = np.linalg.cholesky(S + jitter * np.eye(n_obs))
                break
            except np.linalg.LinAlgError:
                continue
        if L is None:
            # Fallback: use SVD to compute S_inv
            try:
                U, s, Vt = np.linalg.svd(S)
                s_inv = 1.0 / np.maximum(s, 1e-8)
                S_inv = Vt.T @ np.diag(s_inv) @ U.T
            except Exception:
                S_inv = np.eye(n_obs) * 1e-3
    
    # Cross-covariance P_xo (state space to observation space)
    P_xo = (A.T @ A_obs) / (n_members - 1.0)  # (n_vars, n_obs)
    
    # Apply localization to cross-covariance
    P_xo_loc = P_xo * loc_weights
    
    # --- Step 4: Compute Kalman Gain and Stochastic Update ---
    
    if L is not None:
        # Standard Cholesky path
        L_inv = np.linalg.inv(L)
        S_inv = L_inv.T @ L_inv
        K = P_xo_loc @ S_inv
    else:
        K = P_xo_loc @ S_inv
    
    # Update mean
    ens_mean_updated = ens_mean_new + K @ innovation
    
    # Stochastic update using random rotation
    rng = np.random.default_rng(42 + int(time.time() * 1000) % 1000000)
    
    # Generate random vectors for each member
    Z = rng.standard_normal((n_members, n_obs))
    
    if L is not None:
        # Generate noise vectors: v_i = L @ z_i where z_i ~ N(0, I)
        noise = Z @ L.T  # (n_members, n_obs)
    else:
        # Fallback: use uncorrelated noise scaled by sqrt(R)
        noise = Z * np.sqrt(R_val)
    
    # Update anomalies
    A_new = A + noise @ K.T
    
    # Re-center anomalies to ensure the ensemble mean is exactly ens_mean_updated
    A_new = A_new - np.mean(A_new, axis=0)
    
    # Reconstruct ensemble
    ens_updated = ens_mean_updated[None, :] + A_new
    
    # --- Step 5: Post-processing ---
    ens_updated = np.clip(ens_updated, -100.0, 100.0)
    ens_updated = np.where(np.isfinite(ens_updated), ens_updated, 0.0)
    
    if time.time() > deadline:
        return ens_updated.astype(ensemble.dtype)
    
    return ens_updated.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
