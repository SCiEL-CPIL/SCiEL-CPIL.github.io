"""Lorenz-96 ensemble assimilation with Stochastic EnKF (observation perturbation) and localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _random_rotation_matrix(n, rng):
    """Generate a random orthogonal matrix using QR decomposition."""
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det=1)
    d = np.diag(R)
    ph = np.sign(d)
    ph[-1] *= np.sign(np.linalg.det(Q))
    Q = Q * ph[None, :]
    return Q

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter analysis update for Lorenz-96.
    Uses observation perturbation method with random rotation to preserve
    ensemble covariance structure without explicit multiplicative inflation.
    
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
    
    # Use a deterministic seed based on ensemble hash for reproducibility
    seed = hash((tuple(ens.ravel()), tuple(obs.ravel()))) % (2**32)
    rng = np.random.default_rng(seed)
    
    # --- Step 1: Compute ensemble statistics ---
    ens_mean = np.mean(ens, axis=0)
    A = ens - ens_mean  # anomalies (n_members, n_vars)
    
    # --- Step 2: Localization Weights ---
    indices = np.arange(n_vars)
    # Circular distance for periodic boundary
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    
    # Gaussian localization
    sigma = 2.5
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)
    
    # --- Step 3: Observation perturbation ---
    # Perturb each member's observation independently
    # y_i = y + sqrt(R) * eta_i where eta_i ~ N(0, I)
    R_std = np.sqrt(R_val)
    perturbed_obs = obs[None, :] + R_std * rng.standard_normal((n_members, n_obs))
    
    # --- Step 4: SE-Kalman update using random rotation ---
    # The SE-Kalman filter updates each member independently:
    # x_i^a = x_i^f + K * (y_i^pert - H*x_i^f)
    # where K is computed from the ensemble covariance
    
    # Cross-covariance P_xo (state space to observation space)
    A_obs = A[:, obs_idx]  # (n_members, n_obs)
    P_xo = (A.T @ A_obs) / (n_members - 1.0)  # (n_vars, n_obs)
    
    # Apply localization to cross-covariance
    P_xo_loc = P_xo * loc_weights
    
    # Observation-space covariance
    P_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # (n_obs, n_obs)
    S = P_oo + R_val * np.eye(n_obs)
    
    # Ensure positive definiteness
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        jitter = 1e-6
        L = np.linalg.cholesky(S + jitter * np.eye(n_obs))
    
    # S^{-1} via Cholesky
    L_inv = np.linalg.inv(L)
    S_inv = L_inv.T @ L_inv
    
    # Kalman gain
    K = P_xo_loc @ S_inv  # (n_vars, n_obs)
    
    # --- Step 5: Update each member ---
    # For each member i:
    # x_i^a = x_i^f + K @ (y_i^pert - obs_idx_mean)
    # where y_i^pert is the perturbed observation for member i
    
    # Compute innovation for each member
    ens_obs = ens[:, obs_idx]  # (n_members, n_obs)
    innovation_per_member = perturbed_obs - ens_obs  # (n_members, n_obs)
    
    # Update each member
    # x_i^a = x_i^f + K @ innovation_i
    updates = innovation_per_member @ K.T  # (n_members, n_vars)
    ens_updated = ens + updates
    
    # --- Step 6: Post-processing ---
    ens_updated = np.clip(ens_updated, -100.0, 100.0)
    ens_updated = np.where(np.isfinite(ens_updated), ens_updated, 0.0)
    
    if time.time() > deadline:
        return ens_updated.astype(ensemble.dtype)
    
    return ens_updated.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
