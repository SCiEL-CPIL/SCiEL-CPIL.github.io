"""Lorenz-96 ensemble assimilation: local EnAKF with adaptive inflation and localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START
_N = 20
_WINDOW_HALF = 3  # window of 7 variables (i-3 to i+3)


def _gaspari_cohn(d, c):
    """Gaspari-Cohn C2 correlation function."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    m1 = d <= 0.5 * c
    m2 = (d > 0.5 * c) & (d <= 1.5 * c)
    m3 = (d > 1.5 * c) & (d <= 2.5 * c)

    dm = d[m1]
    if len(dm) > 0:
        result[m1] = (-5/4)*(dm/c)**5 + 5*(dm/c)**4 - 10*(dm/c)**3 + 5*dm

    dm = d[m2]
    if len(dm) > 0:
        result[m2] = (1/12)*(dm/c)**5 - (5/4)*(dm/c)**4 + (5/2)*(dm/c)**3 - 5*(dm/c)**2 + (25/12)*(dm/c) + 1/2

    dm = d[m3]
    if len(dm) > 0:
        result[m3] = -(1/16)*(dm/c)**5 + (1/4)*(dm/c)**4 - (15/8)*(dm/c)**3 + (5/2)*(dm/c)**2 - (25/12)*(dm/c) + 105/48

    return result


def _periodic_distance(i, j, N):
    """Minimum distance on a ring of size N."""
    d = abs(i - j)
    return min(d, N - d)


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Adjustment Kalman Filter (EnAKF) update for Lorenz-96.
    
    Uses a deterministic square-root transformation that preserves the ensemble
    mean while adjusting anomalies to match the Kalman-filtered covariance.
    This avoids stochastic noise injection that degrades small ensembles.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations
    observed_indices: array of observed state indices
    observation_variance: scalar observation noise variance
    
    Returns:
    Updated ensemble (M, N)
    """
    t0 = time.time()
    M, N = ensemble.shape
    K = len(observed_indices)
    result = ensemble.astype(float).copy()
    
    # Build observed index lookup
    obs_set = set(int(x) for x in observed_indices)
    obs_map = {int(observed_indices[k]): k for k in range(K)}
    
    # Adaptive multiplicative inflation (slightly more aggressive for EnAKF)
    if M > 2:
        obs_idx_list = list(obs_set)
        spread = np.std(ensemble[:, obs_idx_list], axis=0)
        obs_err = np.sqrt(observation_variance)
        if obs_err > 0:
            ratio = np.mean(spread) / obs_err
        else:
            ratio = 1.0
        if ratio < 1.0:
            inflation = 1.0 + 0.08 * (1.0 - ratio)
        else:
            inflation = 1.003
        inflation = float(np.clip(inflation, 1.003, 1.08))
    else:
        inflation = 1.003
    
    mean = np.mean(result, axis=0)
    result = mean + inflation * (result - mean)
    
    # Correlation length for localization
    corr_len = 4.0
    
    # Precompute localization weights for each center i
    loc_weights_by_i = []
    for i in range(N):
        dists = np.array([_periodic_distance(i, (i + j) % N, N) for j in range(-_WINDOW_HALF, _WINDOW_HALF + 1)])
        w = _gaspari_cohn(dists, corr_len)
        w[_WINDOW_HALF] = 1.0  # ensure center weight is 1
        loc_weights_by_i.append(w)
    
    # Time budget with safety margin
    time_budget = 130.0
    
    for i in range(N):
        if time.time() - t0 > time_budget:
            break
        
        # Local window indices (periodic)
        local_idx = [(i + j) % N for j in range(-_WINDOW_HALF, _WINDOW_HALF + 1)]
        n_local = len(local_idx)
        
        # Find which local positions are observed
        local_obs_pos = []
        for l, g in enumerate(local_idx):
            if g in obs_set:
                local_obs_pos.append(l)
        
        if len(local_obs_pos) == 0:
            continue
        
        n_obs_local = len(local_obs_pos)
        
        # Extract local ensemble: (M, n_local)
        local_ens = result[:, local_idx].copy()
        
        # Local observations
        y_local = np.array([observations[obs_map[local_idx[l]]] for l in local_obs_pos])
        
        # Local mean and anomalies
        local_mean = np.mean(local_ens, axis=0)
        local_anoms = local_ens - local_mean  # (M, n_local)
        
        # Apply localization to anomalies (element-wise)
        w = loc_weights_by_i[i]
        local_anoms_loc = local_anoms * w[np.newaxis, :]
        
        # Local ensemble covariance with localization
        C = (local_anoms_loc.T @ local_anoms_loc) / (M - 1)  # (n_local, n_local)
        
        # Observation operator: (n_obs_local, n_local)
        H = np.zeros((n_obs_local, n_local))
        for k, l in enumerate(local_obs_pos):
            H[k, l] = 1.0
        
        # Innovation covariance S = H C H^T + R
        HC = H @ C  # (n_obs_local, n_local)
        S = HC @ H.T + observation_variance * np.eye(n_obs_local)
        
        # Compute the EnAKF transformation matrix
        # We want to find A such that:
        #   C_anom_new = A @ C_anom @ A.T
        # where C_anom is the localized anomaly covariance
        # and the new anomaly covariance should be (I - K H) C (I - K H)^T
        # but we work in anomaly space.
        
        # The EnAKF approach: compute the gain in anomaly space
        # K_anom = C H^T S^{-1}  (n_local, n_obs_local)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        K_gain = C @ H.T @ S_inv  # (n_local, n_obs_local)
        
        # The update matrix for anomalies: M_upd = (I - K_gain H)
        M_upd = np.eye(n_local) - K_gain @ H  # (n_local, n_local)
        
        # We need to find a square root transformation T such that:
        #   T @ local_anoms_loc @ T.T = M_upd @ C @ M_upd.T
        # Actually, we want: new_anoms = T @ local_anoms_loc
        # such that Cov(new_anoms) = M_upd @ C @ M_upd.T
        
        # Use Cholesky-like approach:
        # C_new_target = M_upd @ C @ M_upd.T
        C_new_target = M_upd @ C @ M_upd.T
        
        # We have C = local_anoms_loc.T @ local_anoms_loc / (M-1)
        # So local_anoms_loc = L @ L.T where L is some factor
        # Actually, let's use SVD to find the transformation
        
        # Let A = local_anoms_loc (M x n_local)
        # We want T (n_local x n_local) such that:
        #   (T @ A).T @ (T @ A) / (M-1) = C_new_target
        #   A.T @ T.T @ T @ A / (M-1) = C_new_target
        
        # Use the fact that C = A.T @ A / (M-1)
        # We want T such that T @ C @ T.T = C_new_target
        # i.e., T @ A.T @ A @ T.T / (M-1) = C_new_target
        
        # Let's use eigen-decomposition approach:
        # C = V_d @ D_d @ V_d.T
        # C_new_target = V_n @ D_n @ V_n.T
        # Then T = V_n @ D_n^{1/2} @ D_d^{-1/2} @ V_d.T
        
        try:
            eigvals_c, eigvecs_c = np.linalg.eigh(C)
            eigvals_new, eigvecs_new = np.linalg.eigh(C_new_target)
            
            # Clip small negative eigenvalues to zero
            eigvals_c = np.clip(eigvals_c, 1e-10, None)
            eigvals_new = np.clip(eigvals_new, 1e-10, None)
            
            # T = V_new @ sqrt(D_new) @ sqrt(D_c)^{-1} @ V_c.T
            sqrt_eig_c = np.sqrt(eigvals_c)
            sqrt_eig_new = np.sqrt(eigvals_new)
            
            # Avoid division by zero
            inv_sqrt_eig_c = 1.0 / np.maximum(sqrt_eig_c, 1e-10)
            
            T = eigvecs_new @ np.diag(sqrt_eig_new * inv_sqrt_eig_c) @ eigvecs_c.T
            
        except np.linalg.LinAlgError:
            # Fallback: use the direct gain-based update
            T = M_upd
        
        # Apply transformation to anomalies
        new_anoms = local_anoms_loc @ T.T  # (M, n_local)
        
        # Update ensemble: new_ens = local_mean + new_anoms
        # But we need to ensure the mean is preserved (it is, since anomalies sum to zero)
        new_local_ens = local_mean[np.newaxis, :] + new_anoms
        
        # Store back
        result[:, local_idx] = new_local_ens
    
    # Safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
# EVOLVE-BLOCK-END
