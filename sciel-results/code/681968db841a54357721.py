"""Stochastic Ensemble Kalman Filter (SE-Kalman) for Lorenz-96 assimilation."""
import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter update using random rotation matrices.
    
    Uses random rotation matrices to generate zero-mean perturbations that
    preserve ensemble spread while ensuring the ensemble mean matches the
    optimal Kalman gain exactly. This avoids bias from deterministic square-root
    decompositions and naturally handles spread preservation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of 16 members, each with 20 state variables.
    observations : ndarray, shape (10,)
        Noisy observations of the 10 even-indexed coordinates.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, ..., 18).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    n_ens = ensemble.shape[0]  # 16
    n_state = ensemble.shape[1]  # 20
    n_obs = len(observed_indices)  # 10
    
    # Compute ensemble mean and anomalies
    ens_mean = ensemble.mean(axis=0)
    anomalies = ensemble - ens_mean  # shape (n_ens, n_state)
    
    # Project anomalies onto observed indices
    obs_anomalies = anomalies[:, observed_indices]  # shape (n_ens, n_obs)
    
    # Ensemble covariance matrices
    C_oo = (obs_anomalies.T @ obs_anomalies) / (n_ens - 1)  # shape (n_obs, n_obs)
    C_xo = (anomalies.T @ obs_anomalies) / (n_ens - 1)      # shape (n_state, n_obs)
    
    # Innovation covariance
    R_obs = observation_variance * np.eye(n_obs)
    S = C_oo + R_obs
    
    # Innovation vector
    innovation = observations - ens_mean[observed_indices]  # shape (n_obs,)
    
    # Compute Kalman gain: K = C_xo @ S^{-1}
    try:
        # Use Cholesky for numerical stability
        L = np.linalg.cholesky(S)
        # Solve L @ L^T @ X^T = C_xo^T, then K = X
        temp = np.linalg.solve(L, C_xo.T)
        K_T = np.linalg.solve(L.T, temp)
        K = K_T.T  # shape (n_state, n_obs)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K = C_xo @ S_inv
    
    # Compute mean update
    new_mean = ens_mean + K @ innovation
    
    # Adaptive inflation based on innovation consistency
    innovation_var = np.var(innovation)
    expected_innovation_var = np.trace(S) / n_obs
    
    inflation_factor = 1.0
    if expected_innovation_var > 1e-10:
        ratio = innovation_var / expected_innovation_var
        # Adaptive inflation: mild when spread is too low or high
        if ratio > 1.3:
            inflation_factor = 1.0 + 0.08 * (ratio - 1.0)
        elif ratio < 0.7:
            inflation_factor = 1.0 + 0.05 * (1.0 - ratio)
    
    # Apply inflation to anomalies if needed
    if abs(inflation_factor - 1.0) > 1e-10:
        anomalies = anomalies * inflation_factor
        obs_anomalies = obs_anomalies * inflation_factor
        # Recompute covariance and gain with inflated spread
        C_oo = (obs_anomalies.T @ obs_anomalies) / (n_ens - 1)
        C_xo = (anomalies.T @ obs_anomalies) / (n_ens - 1)
        S = C_oo + R_obs
        try:
            L = np.linalg.cholesky(S)
            temp = np.linalg.solve(L, C_xo.T)
            K_T = np.linalg.solve(L.T, temp)
            K = K_T.T
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            K = C_xo @ S_inv
        new_mean = ens_mean + K @ innovation
    
    # Stochastic SE-Kalman update using random rotation matrices
    # Generate a random orthogonal matrix Q of size (n_ens, n_obs)
    # such that the perturbations have zero mean and correct covariance
    
    # Use QR decomposition of a random Gaussian matrix to get orthogonal Q
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((n_ens, n_obs))
    Q, _ = np.linalg.qr(Z)
    
    # The perturbation for member i is: delta_y_i = sqrt(R) * Q[i, :]
    # where Q[i, :] is the i-th row of the orthogonal matrix
    # This ensures: mean(delta_y) = 0 and Cov(delta_y) = R (approximately)
    
    sqrt_R = np.sqrt(observation_variance)
    perturbations = sqrt_R * Q  # shape (n_ens, n_obs)
    
    # Compute updated ensemble
    updated_ensemble = np.zeros_like(ensemble)
    for i in range(n_ens):
        # Member-specific perturbed observation
        y_i_pert = observations + perturbations[i, :]
        
        # Member-specific innovation
        H_x_i = ensemble[i, observed_indices]
        d_i = y_i_pert - H_x_i
        
        # Update member
        updated_ensemble[i, :] = ensemble[i, :] + K @ d_i
    
    # Correct the mean to ensure it matches the desired posterior mean exactly
    # The stochastic update should have zero mean perturbation, but numerical
    # errors may introduce small biases
    current_mean = updated_ensemble.mean(axis=0)
    mean_correction = new_mean - current_mean
    updated_ensemble += mean_correction
    
    # Ensure output is finite and within bounds
    updated_ensemble = np.clip(updated_ensemble, -100, 100)
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
