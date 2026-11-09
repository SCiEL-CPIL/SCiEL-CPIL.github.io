"""Lorenz-96 ensemble assimilation with Stochastic EnKF."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter (SE-Kalman) analysis update for Lorenz-96.
    Uses stochastic perturbations to preserve ensemble spread and avoid numerical
    instabilities associated with deterministic square-root filters.
    
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
    
    # 1. Compute ensemble mean and anomalies
    ens_mean = np.mean(ens, axis=0)
    A = ens - ens_mean  # Shape (n_members, n_vars)
    A_obs = A[:, obs_idx]  # Shape (n_members, n_obs)
    
    # 2. Compute covariances
    # Cross-covariance P_xo (n_vars x n_obs)
    P_xo = (A.T @ A_obs) / (n_members - 1.0)
    # Observation covariance P_oo (n_obs x n_obs)
    P_oo = (A_obs.T @ A_obs) / (n_members - 1.0)
    
    # 3. Innovation
    innovation = obs - ens_mean[obs_idx]  # Shape (n_obs,)
    
    # 4. Compute Kalman Gain
    # S = P_oo + R
    S = P_oo + R_val * np.eye(n_obs)
    # Add jitter for numerical stability
    S += 1e-6 * np.eye(n_obs)
    
    try:
        # K = P_xo @ inv(S)
        # Solve S^T X^T = P_xo^T => X = P_xo @ inv(S)
        # Since S is symmetric, solve(S, P_xo.T).T
        K = np.linalg.solve(S, P_xo.T).T  # Shape (n_vars, n_obs)
    except np.linalg.LinAlgError:
        # Fallback to pinv if solve fails
        K = P_xo @ np.linalg.pinv(S)
    
    # 5. Update Mean
    ens_mean_new = ens_mean + K @ innovation
    
    # 6. Update Anomalies (Stochastic EnKF)
    # For each member i, the update is:
    # A_new[i] = A[i] - K @ (A_obs[i] + epsilon_i)
    # where epsilon_i ~ N(0, R)
    # This can be vectorized.
    # A_obs shape: (n_members, n_obs)
    # K shape: (n_vars, n_obs)
    # We need K @ (A_obs[i] + eps[i]) for all i.
    # Let Y_pert = A_obs + E, where E is (n_members, n_obs) noise.
    # Then A_new = A - Y_pert @ K.T
    
    # Generate stochastic perturbations
    # R is scalar variance, so each component is independent N(0, R_val)
    E = np.random.normal(0.0, np.sqrt(R_val), size=(n_members, n_obs))
    
    Y_pert = A_obs + E  # Shape (n_members, n_obs)
    
    # Update anomalies
    # A_new[i] = A[i] - Y_pert[i] @ K.T
    # Vectorized: A_new = A - Y_pert @ K.T
    A_new = A - Y_pert @ K.T
    
    # Reconstruct ensemble
    ens = ens_mean_new[None, :] + A_new
    
    # 7. Final safety checks
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
        
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
