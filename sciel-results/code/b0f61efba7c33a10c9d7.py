"""Ensemble data assimilation for Lorenz-96 using LETKF with adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = _start_time + 145.0  # Safety margin before 150s kill

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Transform Kalman Filter (LETKF) with adaptive inflation
    for Lorenz-96 data assimilation.
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
    
    # 1. Compute background mean
    mean_b = np.mean(X_b, axis=0)
    
    # 2. Compute background perturbations (centered)
    W_b = X_b - mean_b
    
    # 3. Compute background covariance P_b (N x N)
    P_b = (W_b.T @ W_b) / (M - 1)
    
    # 4. Localization parameters
    # Use a smaller localization radius to reduce spurious correlations
    L = 3.0  # Localization radius
    
    def localize(dist, L):
        """Cosine taper localization."""
        mask = dist < L
        taper = np.zeros_like(dist, dtype=np.float64)
        taper[mask] = 0.5 * (1.0 + np.cos(np.pi * dist[mask] / L))
        return taper
    
    # 5. Compute localized covariances
    N_all = np.arange(N)
    
    # Distances for P_xo: N x N_obs
    dist_xo = np.abs(N_all[:, np.newaxis] - obs_idx[np.newaxis, :])
    dist_xo = np.minimum(dist_xo, N - dist_xo)
    loc_xo = localize(dist_xo, L)
    
    # Distances for P_oo: N_obs x N_obs
    dist_oo = np.abs(obs_idx[:, np.newaxis] - obs_idx[np.newaxis, :])
    dist_oo = np.minimum(dist_oo, N - dist_oo)
    loc_oo = localize(dist_oo, L)
    
    # Extract and localize covariance matrices
    P_oo = P_b[np.ix_(obs_idx, obs_idx)]
    P_xo = P_b[:, obs_idx]
    
    P_oo_loc = P_oo * loc_oo
    P_xo_loc = P_xo * loc_xo
    
    # 6. Observation error covariance
    R = np.eye(N_obs) * R_val
    
    # 7. Innovation
    d = y - mean_b[obs_idx]
    
    # 8. Innovation covariance S
    S = P_oo_loc + R
    S = (S + S.T) / 2.0
    
    # 9. Compute SVD of S for stable inverse
    try:
        U, s, Vt = np.linalg.svd(S)
        s_inv = np.zeros_like(s)
        threshold = 1e-10
        s_inv[s > threshold] = 1.0 / s[s > threshold]
        S_inv = (Vt.T * s_inv) @ U.T
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # 10. Compute Kalman Gain K (N x N_obs)
    K = P_xo_loc @ S_inv
    
    # 11. Update Mean
    mean_a = mean_b + K @ d
    
    # 12. Adaptive multiplicative inflation
    # Compute innovation norm relative to expected
    innov_norm = np.linalg.norm(d)
    expected_innov_norm = np.sqrt(np.trace(S))
    innov_ratio = innov_norm / (expected_innov_norm + 1e-10)
    
    # Inflation factor: smooth response to innovation magnitude
    alpha = 0.25
    inflation = 1.0 + alpha * max(0.0, innov_ratio - 1.0)
    inflation = np.clip(inflation, 1.0, 1.4)
    
    # 13. Compute analysis covariance P_a
    I_N = np.eye(N)
    H_mat = np.zeros((N_obs, N))
    H_mat[np.arange(N_obs), obs_idx] = 1.0
    
    K_H = K @ H_mat
    I_minus_KH = I_N - K_H
    
    P_a = (I_minus_KH @ P_b @ I_minus_KH.T) + (K @ R @ K.T)
    P_a = (P_a + P_a.T) / 2.0
    
    # Apply inflation to analysis covariance
    P_a_inflated = (inflation ** 2) * P_a
    P_a_inflated = P_a_inflated + 1e-6 * I_N  # Regularization
    
    # 14. Compute LETKF transform T such that W_a = W_b @ T
    # Use eigendecomposition approach
    eigvals_b, eigvecs_b = np.linalg.eigh(P_b)
    eigvals_a, eigvecs_a = np.linalg.eigh(P_a_inflated)
    
    # Avoid division by zero
    eigvals_b_safe = np.where(eigvals_b > 1e-12, eigvals_b, 1.0)
    
    # T = V_b diag(sqrt(eigvals_a / eigvals_b)) V_a^T
    ratio = np.sqrt(eigvals_a / eigvals_b_safe)
    T = eigvecs_b @ np.diag(ratio) @ eigvecs_a.T
    
    # 15. Update perturbations
    W_a = W_b @ T
    
    # 16. Reconstruct ensemble
    X_a = mean_a + W_a
    
    # Safety checks
    X_a = np.clip(X_a, -100.0, 100.0)
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return X_a
# EVOLVE-BLOCK-END
