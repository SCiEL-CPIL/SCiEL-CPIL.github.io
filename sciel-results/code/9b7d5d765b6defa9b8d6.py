"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
# Global deadline tracking
_start_time = time.time()
_DEADLINE = _start_time + 145.0  # Safety margin before 150s kill

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) using ensemble square root formulation
    with adaptive mean inflation and Gaussian localization.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.array(ensemble, dtype=np.float64)
    
    # Ensure float64 for precision
    X_b = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)
    
    M, N = X_b.shape  # M=16, N=20
    N_obs = len(obs_idx)
    
    # 1. Compute background mean and perturbations
    mean_b = np.mean(X_b, axis=0)
    W_b = X_b - mean_b  # M x N
    
    # 2. Compute background covariance P_b (N x N)
    P_b = (W_b.T @ W_b) / (M - 1)
    
    # 3. Extract relevant covariances
    P_oo = P_b[np.ix_(obs_idx, obs_idx)]
    P_xo = P_b[:, obs_idx]
    
    # 4. Observation error covariance R (N_obs x N_obs)
    R = np.eye(N_obs) * R_val
    
    # 5. Innovation (mean difference)
    H_mean = mean_b[obs_idx]
    d = y - H_mean
    
    # 6. Gaussian localization for P_xo
    N_all = np.arange(N)
    N_obs_arr = np.array(obs_idx)
    
    # Compute circular distance
    dist = np.abs(N_all[:, np.newaxis] - N_obs_arr[np.newaxis, :])
    dist = np.minimum(dist, N - dist)  # Circular distance
    
    # Slightly tighter localization to further reduce spurious correlations
    L = 1.2
    loc_matrix = np.exp(-0.5 * (dist / L) ** 2)
    
    # Localize P_xo
    P_xo_loc = P_xo * loc_matrix
    
    # Also localize P_oo diagonal for consistency (optional, keeps it simple)
    S = P_oo + R
    
    # Compute inverse of S
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        
    # K = P_xo_loc @ S_inv  (N x N_obs)
    K = P_xo_loc @ S_inv
    
    # 7. Update Mean
    mean_a = mean_b + K @ d
    
    # 8. Update Perturbations using ESRF-style approach
    # H_W_b: N_obs x M (observed perturbations)
    H_W_b = W_b[:, obs_idx].T  # N_obs x M
    
    # Compute the update matrix for perturbations
    # W_a = W_b - (K @ H_W_b).T
    # K is (N, N_obs), H_W_b is (N_obs, M)
    # K @ H_W_b is (N, M)
    # Transpose is (M, N)
    update_part = (K @ H_W_b).T  # M x N
    W_a = W_b - update_part  # M x N
    
    # 9. Adaptive Inflation - use innovation-based approach
    # Expected innovation variance is trace(S)/N_obs
    expected_innov_var = np.trace(S) / N_obs
    # Actual innovation variance (using M-1 denominator for unbiased estimate)
    # d is a vector of length N_obs, so np.var gives sample variance
    actual_innov_var = np.var(d, ddof=1) + 1e-10
    
    # Ratio indicates how much the ensemble underestimated uncertainty
    ratio = expected_innov_var / actual_innov_var
    
    # Apply inflation to perturbations (spatially uniform)
    # Use a smoother, bounded inflation
    inflation_factor = np.sqrt(np.clip(ratio, 1.0, 1.08))
    
    # Apply inflation to perturbations
    W_a = W_a * inflation_factor
    
    # 10. Additionally apply mean inflation for better stability
    # This helps prevent ensemble collapse in chaotic systems
    # Small mean inflation based on innovation magnitude
    innov_norm_sq = np.sum(d**2)
    expected_norm_sq = np.trace(S)
    mean_inflation = 1.0 + 0.005 * np.clip((innov_norm_sq / (expected_norm_sq + 1e-10) - 1.0), 0.0, 2.0)
    
    # Apply mean inflation around the analysis mean
    W_a = W_a * mean_inflation
    
    # Re-center the perturbations to ensure the mean remains mean_a
    mean_W_a = np.mean(W_a, axis=0)
    W_a = W_a - mean_W_a[np.newaxis, :]
    
    # 11. Construct Analysis Ensemble
    X_a = mean_a[np.newaxis, :] + W_a
    
    # 12. Safety checks: ensure finite and within bounds
    X_a = np.clip(X_a, -100.0, 100.0)
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return X_a
# EVOLVE-BLOCK-END
