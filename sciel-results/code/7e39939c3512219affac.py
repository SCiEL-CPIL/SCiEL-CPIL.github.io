"""Lorenz-96 assimilation using a localized Particle Flow Filter with adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96 system. x has shape (N, 20) or (20,)."""
    if x.ndim == 1:
        x_1 = np.roll(x, 1)
        x_m1 = np.roll(x, -1)
        x_m2 = np.roll(x, 2)
        return (x_1 - x_m2) * x_m1 - x + F
    else:
        x_1 = np.roll(x, 1, axis=1)
        x_m1 = np.roll(x, -1, axis=1)
        x_m2 = np.roll(x, 2, axis=1)
        return (x_1 - x_m2) * x_m1 - x + F


def _circular_distance(i, j, n):
    """Compute circular distance between indices i and j on a ring of size n."""
    d = abs(i - j)
    return min(d, n - d)


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a localized Particle Flow Filter analysis update with adaptive inflation.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Make a copy to work with
    result = ensemble.copy()
    
    # Compute ensemble mean and anomalies
    ens_mean = np.mean(result, axis=0)
    anomalies = result - ens_mean
    
    # Compute sample covariance (N-1 normalization)
    cov = (anomalies.T @ anomalies) / (n_members - 1)
    
    # Observation operator: select observed indices
    H = np.zeros((n_obs, n_state))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Observation error covariance (diagonal)
    R = np.eye(n_obs) * observation_variance
    
    # Observation innovation
    innovation = observations - H @ ens_mean
    
    # Innovation covariance: S = H P H^T + R
    HP = H @ cov
    S = HP @ H.T + R
    
    # Kalman gain: K = P H^T S^{-1}
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = (cov @ H.T) @ S_inv  # Shape: (n_state, n_obs)
    
    # Apply localization to the Kalman gain
    # Use a Gaussian decay based on circular distance
    decay_length = 2.5
    
    localized_K = np.zeros_like(K)
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        for i in range(n_state):
            # Circular distance
            d = _circular_distance(i, obs_idx, n_state)
            w = np.exp(-d**2 / (2 * decay_length**2))
            localized_K[i, j] = K[i, j] * w
    
    # Compute the posterior mean and covariance
    posterior_mean = ens_mean + localized_K @ innovation
    
    # Posterior covariance: P_a = P_f - K H P_f (Joseph form for stability)
    # Using: P_a = (I - K H) P_f (I - K H)^T + K R K^T
    I_KH = np.eye(n_state) - localized_K @ H
    posterior_cov = I_KH @ cov @ I_KH.T + localized_K @ R @ localized_K.T
    
    # Ensure symmetry
    posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)
    
    # Particle Flow Filter: map prior particles to posterior distribution
    # using Cholesky decomposition for a proper linear transformation
    
    # Add small regularization for numerical stability
    reg = 1e-10 * np.eye(n_state)
    
    try:
        L_prior = np.linalg.cholesky(cov + reg)
        L_posterior = np.linalg.cholesky(posterior_cov + reg)
    except np.linalg.LinAlgError:
        # Fallback to eigenvalue decomposition
        evals_prior, evecs_prior = np.linalg.eigh(cov + reg)
        evals_prior = np.maximum(evals_prior, 0)
        L_prior = evecs_prior @ np.diag(np.sqrt(evals_prior))
        
        evals_post, evecs_post = np.linalg.eigh(posterior_cov + reg)
        evals_post = np.maximum(evals_post, 0)
        L_posterior = evecs_post @ np.diag(np.sqrt(evals_post))
    
    # Transform prior particles to standard normal space
    # z_m = L_prior^{-1} (x_m - ens_mean)
    try:
        L_prior_inv = np.linalg.inv(L_prior)
    except np.linalg.LinAlgError:
        L_prior_inv = np.linalg.pinv(L_prior)
    
    # Standardize the particles
    z_particles = np.zeros_like(anomalies)
    for m in range(n_members):
        z_particles[m] = L_prior_inv @ anomalies[m]
    
    # Transform back to posterior space
    # x_a_m = posterior_mean + L_posterior @ z_m
    for m in range(n_members):
        result[m] = posterior_mean + L_posterior @ z_particles[m]
    
    # Adaptive inflation to prevent filter collapse
    # Use a smooth, stable inflation scheme based on innovation ratio
    
    # Estimate expected innovation variance
    S_expected = HP @ H.T + R
    S_expected_diag = np.diag(S_expected)
    
    # Actual innovation variance per observation
    innov_var_per_obs = np.var(innovation)
    
    # Compute inflation factor using mean log-ratio for stability
    if np.any(S_expected_diag > 1e-10):
        valid_mask = S_expected_diag > 1e-10
        if np.any(valid_mask):
            ratios = innov_var_per_obs / S_expected_diag[valid_mask]
            ratios = np.clip(ratios, 0.3, 3.0)
            log_ratio = np.mean(np.log(ratios))
            # Dampen the inflation adjustment for stability
            inflation = np.exp(np.clip(0.4 * log_ratio, -0.2, 0.2))
        else:
            inflation = 1.0
    else:
        inflation = 1.0
    
    # Apply inflation around the analysis mean
    new_mean = np.mean(result, axis=0)
    result = new_mean + inflation * (result - new_mean)
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    
    # Ensure finiteness
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return best result found so far
        pass
    
    return result

# EVOLVE-BLOCK-END
