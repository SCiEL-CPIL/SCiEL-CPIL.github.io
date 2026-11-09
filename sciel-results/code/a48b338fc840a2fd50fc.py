"""Lorenz-96 ensemble assimilation with Stochastic EnKF, adaptive inflation, and localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter analysis update for Lorenz-96.
    Uses random rotation for stochastic perturbations, adaptive inflation, and Gaussian localization.
    
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
    target_spread = np.sqrt(R_val) * 1.2
    
    # Inflation factor using log-sum-exp style soft clamp
    ratio = target_spread / np.maximum(current_spread, 1e-15)
    # Use a smooth soft-max to avoid discontinuities
    log_ratios = np.log(np.maximum(ratio, 1e-10))
    max_log = np.max(log_ratios)
    log_sum_exp = max_log + np.log(np.sum(np.exp(log_ratios - max_log)))
    soft_mean_ratio = np.exp(log_sum_exp - np.log(len(ratio)))
    
    raw_inflation = np.clip(soft_mean_ratio, 0.99, 1.20)
    
    # Blend with base inflation
    base_inflation = 1.01
    inflation = 0.4 * base_inflation + 0.6 * raw_inflation
    inflation = np.clip(inflation, 1.0, 1.18)
    
    # Apply multiplicative inflation
    ens = ens_mean + inflation * ens_centered
    
    # --- Step 2: Localization Weights ---
    indices = np.arange(n_vars)
    # Circular distance for periodic boundary
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    
    # Gaussian localization with narrower sigma for reduced spurious correlations
    sigma = 2.2
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
            # Fallback: use SVD or regularized solve if Cholesky fails completely
            # We can approximate S_inv via SVD
            try:
                U, s, Vt = np.linalg.svd(S)
                s_inv = 1.0 / np.maximum(s, 1e-8)
                S_inv = Vt.T @ np.diag(s_inv) @ U.T
                # For the stochastic update, we need a matrix Q such that Q Q^T = S
                # Or we can just use the gain directly with a simpler stochastic scheme
                # Let's stick to a robust solve for the gain and use a simple random perturbation
                # K = P_xo S^-1
                # We'll compute K later.
                pass
            except Exception:
                # Absolute fallback: identity gain scaled small
                S_inv = np.eye(n_obs) * 1e-3
                L = np.eye(n_obs) # Dummy, not used in this branch
    
    # Cross-covariance P_xo (state space to observation space)
    P_xo = (A.T @ A_obs) / (n_members - 1.0)  # (n_vars, n_obs)
    
    # Apply localization to cross-covariance
    P_xo_loc = P_xo * loc_weights
    
    # --- Step 4: Compute Kalman Gain and Stochastic Update ---
    
    if L is not None:
        # Standard Cholesky path
        # Solve L X = P_xo_loc^T -> X = L^-1 P_xo_loc^T
        # Then K = X^T L^-T
        # Alternatively: S_inv = L^-T L^-1
        L_inv = np.linalg.inv(L)
        S_inv = L_inv.T @ L_inv
        K = P_xo_loc @ S_inv
    else:
        # Fallback path using SVD-computed S_inv or dummy
        # If S_inv was computed via SVD in the except block above, use it
        # Note: The variable S_inv might be defined in the try block inside the except
        # To be safe, re-calculate if S_inv is not defined
        if 'S_inv' not in locals():
             S_inv = np.eye(n_obs) * 1e-3
        K = P_xo_loc @ S_inv

    # Update mean
    ens_mean_updated = ens_mean_new + K @ innovation
    
    # Stochastic update:
    # We use the perturbed observation form.
    # Generate a random rotation matrix Q (orthogonal)
    rng = np.random.default_rng(42 + int(time.time() * 1000) % 1000000)
    
    # Generate random vectors for each member
    Z = rng.standard_normal((n_members, n_obs))
    
    # If we have a valid Cholesky factor L, we can use it to generate correlated noise
    # However, the standard stochastic EnKF uses independent perturbations in the observation space
    # projected through the gain.
    # x_i_a = x_mean_a + (x_i_f - x_mean_f) + K @ (y_i_pert - y_mean_obs)
    # where y_i_pert - y_mean_obs are i.i.d N(0, S)
    
    # To generate N(0, S) vectors:
    # If L is available: v_i = L @ z_i where z_i ~ N(0, I)
    # Then y_i_pert - y_mean = v_i
    
    if L is not None:
        # Generate noise vectors
        noise = Z @ L.T # (n_members, n_obs)
    else:
        # Fallback: use uncorrelated noise scaled by sqrt(R) as a rough approximation
        # This is less optimal but prevents crash
        noise = Z * np.sqrt(R_val)

    # Update anomalies
    # A_new = A + noise @ K.T
    A_new = A + noise @ K.T
    
    # Reconstruct ensemble
    ens_updated = ens_mean_updated[None, :] + A_new
    
    # --- Step 5: Post-processing ---
    ens_updated = np.clip(ens_updated, -100.0, 100.0)
    ens_updated = np.where(np.isfinite(ens_updated), ens_updated, 0.0)
    
    if time.time() > deadline:
        return ens_updated.astype(ensemble.dtype)
    
    return ens_updated.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
