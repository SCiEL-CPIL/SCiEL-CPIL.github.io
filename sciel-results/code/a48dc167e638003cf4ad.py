import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Square Root Filter (EnSRF) with localization.
    This approach uses a deterministic update for the mean and a square-root 
    update for the anomalies to preserve ensemble spread more accurately 
    than the stochastic EnKF.
    """
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=int)
    
    n_members, n_state = ensemble.shape
    n_obs = len(observed_indices)
    
    # 1. Background mean and anomalies
    mean_b = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean_b
    
    # 2. Observation anomalies
    obs_anomalies = anomalies[:, observed_indices] # (N, M)
    
    # 3. Compute Cross-Covariance C_xy (L x M)
    # C_xy[i, j] = Cov(x_i, y_j)
    C_xy = np.zeros((n_state, n_obs))
    for i in range(n_state):
        for j_idx, j in enumerate(observed_indices):
            C_xy[i, j_idx] = np.mean(anomalies[:, i] * obs_anomalies[:, j_idx])
            
    # 4. Localization
    # Localize C_xy using exponential kernel
    loc_length = 3.0
    C_xy_loc = np.zeros_like(C_xy)
    for i in range(n_state):
        for j_idx, j in enumerate(observed_indices):
            dist = abs(i - j)
            weight = np.exp(-dist / loc_length)
            C_xy_loc[i, j_idx] = weight * C_xy[i, j_idx]
            
    # 5. Compute Observation Covariance C_yy (M x M)
    # C_yy = (1/(N-1)) Y^T Y + R
    C_yy = (obs_anomalies.T @ obs_anomalies) / (n_members - 1)
    C_yy += observation_variance * np.eye(n_obs)
    
    # 6. Compute Kalman Gain K (L x M)
    # K = C_xy_loc @ inv(C_yy)
    try:
        K = C_xy_loc @ np.linalg.inv(C_yy)
    except np.linalg.LinAlgError:
        K = C_xy_loc @ np.linalg.pinv(C_yy)
        
    # 7. Update Mean
    innovation = observations - mean_b[observed_indices]
    mean_a = mean_b + K @ innovation
    
    # 8. Update Anomalies using Square Root
    # We want A_a such that A_a A_a^T = A_b A_b^T - A_b K^T K A_b^T (approx)
    # Actually, the exact update for the covariance is C_aa = C_bb - C_xy_loc C_yy^{-1} C_xy_loc^T
    # But the square root update on the anomalies uses the matrix U derived from the gain.
    # U = I - (1/(N-1)) A_b^T K^T K A_b
    # Note: K here is the full gain.
    
    # Compute K^T K (L x L)
    KtK = K.T @ K
    
    # Compute A_b^T K^T K A_b (N x N)
    # A_b is (N, L). KtK is (L, L).
    # tmp = A_b @ KtK (N, L)
    # U = I - (1/(N-1)) tmp @ A_b.T
    tmp = anomalies @ KtK
    U = np.eye(n_members) - (tmp @ anomalies.T) / (n_members - 1)
    
    # Ensure U is symmetric
    U = (U + U.T) / 2.0
    
    # Compute Cholesky decomposition of U
    # U = R^T R
    try:
        R = np.linalg.cholesky(U)
        # U = R^T R, so R is upper triangular.
        # We want A_a = A_b R^T ?
        # Check: A_a A_a^T = A_b R^T R A_b^T = A_b U A_b^T. Correct.
        updated_anomalies = anomalies @ R.T
    except np.linalg.LinAlgError:
        # If Cholesky fails, use eigenvalue decomposition and clip negative eigenvalues
        eigvals, eigvecs = np.linalg.eigh(U)
        eigvals = np.clip(eigvals, 0, None)
        R = eigvecs @ np.diag(np.sqrt(eigvals))
        # U = R^T R?
        # If U = V D V^T, then sqrt(U) = V D^0.5 V^T.
        # We want M such that M^T M = U.
        # M = D^0.5 V^T.
        # Then A_a = A_b M.
        M = np.diag(np.sqrt(eigvals)) @ eigvecs.T
        updated_anomalies = anomalies @ M

    # 9. Reconstruct Ensemble
    updated_ensemble = mean_a + updated_anomalies
    
    # 10. Adaptive Inflation
    # Check spread at observed locations
    obs_spread = np.std(updated_anomalies[:, observed_indices], axis=0)
    spread_mean = np.mean(obs_spread)
    
    inflation_factor = 1.0
    if spread_mean < 1e-6:
        inflation_factor = 1.1
    else:
        # Simple inflation if innovation is large
        innovation_std = np.std(innovation)
        if innovation_std > 2.0 * spread_mean:
            inflation_factor = 1.05
            
    updated_ensemble = mean_a + inflation_factor * (updated_ensemble - mean_a)
    
    # Ensure finiteness and bounds
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
