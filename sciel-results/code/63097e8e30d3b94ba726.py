"""Combined Local Deterministic ESRF analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _gaspari_cohn(distance, L1=1.0, L2=2.0, L3=3.0, L4=4.0, L5=5.0):
    """
    Gaspari-Cohn correlation function for localization.
    
    Parameters:
    distance: array of distances
    L1-L5: shape parameters (default: 1,2,3,4,5)
    
    Returns:
    correlation values in [0, 1]
    """
    d = np.abs(distance)
    result = np.zeros_like(d, dtype=float)
    
    # Region 1: d <= L1
    mask1 = d <= L1
    if np.any(mask1):
        d1 = d[mask1]
        c1 = 1.0 / (12.0 * L1**5)
        result[mask1] = (
            -0.25 * c1 * d1**5
            + 0.5 * c1 * d1**4
            + (5.0/3.0) * c1 * L1 * d1**3
            - (5.0/3.0) * c1 * L1**2 * d1**2
            + 1.0
        )
    
    # Region 2: L1 < d <= L2
    mask2 = (d > L1) & (d <= L2)
    if np.any(mask2):
        d2 = d[mask2]
        c2 = 1.0 / (L2 - L1) * (
            -1.0 / (2.0 * L1**4) + 1.0 / (L1**3) - 5.0/(6.0 * L1**2) + 5.0/(6.0 * L1)
            - 1.0 + 5.0/6.0
        )
        # Simplified form from Gaspari & Cohn (1999)
        result[mask2] = (
            -0.5 * d2**5 / (L2 - L1)
            + 2.5 * L1 * d2**4 / (L2 - L1)
            - 10.0 * L1**2 * d2**3 / (L2 - L1)
            + 12.5 * L1**3 * d2**2 / (L2 - L1)
            - 25.0 * L1**4 * d2 / (L2 - L1)
            + 12.5 * L1**5 / (L2 - L1)
        )
    
    # Region 3: L2 < d <= L3
    mask3 = (d > L2) & (d <= L3)
    if np.any(mask3):
        d3 = d[mask3]
        result[mask3] = (
            d3**5 / (L2 - L3) * (1.0/L2**4 - 1.0/L3**4)
            - 5.0 * d3**4 / (L2 - L3) * (1.0/L2**3 - 1.0/L3**3)
            + 10.0 * d3**3 / (L2 - L3) * (1.0/L2**2 - 1.0/L3**2)
            - 10.0 * d3**2 / (L2 - L3) * (1.0/L2 - 1.0/L3)
            + 5.0 * d3 / (L2 - L3) * (1.0 - 1.0)
            + (
                5.0 * L2**4 / (L2 - L3) - 10.0 * L2**3 / (L2 - L3)
                + 10.0 * L2**2 / (L2 - L3) - 5.0 * L2 / (L2 - L3)
            )
        )
    
    # Region 4: L3 < d <= L4
    mask4 = (d > L3) & (d <= L4)
    if np.any(mask4):
        d4 = d[mask4]
        result[mask4] = (
            -d4**5 / (L4 - L3)
            + 2.5 * L3 * d4**4 / (L4 - L3)
            - 10.0 * L3**2 * d4**3 / (L4 - L3)
            + 12.5 * L3**3 * d4**2 / (L4 - L3)
            - 25.0 * L3**4 * d4 / (L4 - L3)
            + 12.5 * L3**5 / (L4 - L3)
        )
    
    # Region 5: L4 < d <= L5
    mask5 = (d > L4) & (d <= L5)
    if np.any(mask5):
        d5 = d[mask5]
        c5 = 1.0 / (L5 - L4)
        result[mask5] = (
            c5 * (
                -0.25 * d5**5
                + 0.5 * L4 * d5**4
                - (5.0/6.0) * L4**2 * d5**3
                + (5.0/6.0) * L4**3 * d5**2
                - (1.0/2.0) * L4**4 * d5
                + (1.0/12.0) * L4**5
            )
        )
    
    # Beyond L5: zero
    result[d > L5] = 0.0
    
    return np.clip(result, 0.0, 1.0)

def _local_esrf_update(local_ens, local_obs_mask, y_local, obs_var):
    """
    Perform a local Deterministic Ensemble Square Root Filter update.
    
    Parameters:
    local_ens: (M, L) array of local ensemble states
    local_obs_mask: (L,) boolean array indicating which local variables are observed
    y_local: (n_obs,) array of local observations
    obs_var: scalar observation variance
    
    Returns:
    updated local ensemble (M, L)
    """
    M, L = local_ens.shape
    n_obs = len(y_local)
    
    if n_obs == 0:
        return local_ens.copy()
    
    # Identify observed indices in the local window
    obs_idx_in_local = np.where(local_obs_mask)[0]
    
    if len(obs_idx_in_local) == 0:
        return local_ens.copy()
    
    # Extract observed portion
    x_obs = local_ens[:, obs_idx_in_local]  # (M, n_obs)
    x_mean_obs = np.mean(x_obs, axis=0)  # (n_obs,)
    
    # Compute innovation
    innovation = y_local - x_mean_obs  # (n_obs,)
    
    # Compute local covariance for observed variables
    A_obs = x_obs - x_mean_obs[None, :]  # (M, n_obs)
    P_obs = (A_obs.T @ A_obs) / (M - 1)  # (n_obs, n_obs)
    
    # Innovation covariance
    R = obs_var * np.eye(n_obs)
    S = P_obs + R  # (n_obs, n_obs)
    
    # Compute Kalman gain for observed variables
    try:
        # K_obs = P_obs @ S^{-1}, shape (n_obs, n_obs)
        S_inv = np.linalg.inv(S)
        K_obs = P_obs @ S_inv
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K_obs = P_obs @ S_inv
    
    # Update mean for observed variables
    x_mean_obs_analysis = x_mean_obs + K_obs @ innovation  # (n_obs,)
    
    # Update anomalies for observed variables
    # For deterministic ESRF, we need an orthogonal transform T such that:
    # (A_obs - A_obs @ K_obs) @ T = A_obs_analysis
    # This ensures exact covariance matching
    
    # Compute the pre-transform anomalies
    A_obs_pre = A_obs - A_obs @ K_obs  # (M, n_obs)
    
    # Target covariance for analysis
    P_analysis_obs = P_obs - P_obs @ K_obs  # (n_obs, n_obs)
    
    # Find orthogonal transform T such that A_obs_pre @ T has covariance P_analysis_obs
    # Using SVD: A_obs_pre = U1 @ S1 @ V1.T
    # We want (U1 @ S1 @ V1.T) @ T = U2 @ S2 @ V2.T where U2@S2@V2.T has cov P_analysis_obs
    # Simple approach: T = V1.T @ V2 where V2 comes from SVD of target
    
    try:
        # SVD of pre-transform anomalies
        U1, S1, V1T = np.linalg.svd(A_obs_pre, full_matrices=False)
        
        # Target: we want analysis anomalies with covariance P_analysis_obs
        # Use Cholesky: P_analysis_obs = L @ L.T
        L_cholesky = np.linalg.cholesky(P_analysis_obs + 1e-10 * np.eye(n_obs))
        
        # We want A_analysis = A_obs_pre @ T where cov(A_analysis) = P_analysis_obs
        # If A_obs_pre = U1 @ diag(S1) @ V1T, and we set T = V1 @ diag(alpha) @ V2T
        # This gets complex. Simpler: use the fact that for ESRF, we can compute T directly
        
        # Alternative simpler approach: compute T via eigendecomposition
        # P_pre = A_obs_pre.T @ A_obs_pre / (M-1)
        P_pre = (A_obs_pre.T @ A_obs_pre) / (M - 1)
        
        # We want T such that P_pre @ T.T @ T = P_analysis_obs
        # Eigendecompose both
        eigvals_pre, eigvecs_pre = np.linalg.eigh(P_pre)
        eigvals_ana, eigvecs_ana = np.linalg.eigh(P_analysis_obs)
        
        # T = eigvecs_pre @ diag(sqrt(eigvals_ana / eigvals_pre)) @ eigvecs_ana.T
        ratio = np.sqrt(eigvals_ana / (eigvals_pre + 1e-10))
        T = eigvecs_pre @ np.diag(ratio) @ eigvecs_ana.T
        
    except Exception:
        # Fallback: use identity (no anomaly update, only mean update)
        T = np.eye(n_obs)
    
    A_obs_analysis = A_obs_pre @ T  # (M, n_obs)
    
    # Construct updated observed portion
    local_ens_analysis = local_ens.copy()
    local_ens_analysis[:, obs_idx_in_local] = x_mean_obs_analysis[None, :] + A_obs_analysis
    
    return local_ens_analysis

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Deterministic Ensemble Square Root Filter (ESRF) update for Lorenz-96 assimilation.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations for observed_indices
    observed_indices: array of indices that are observed
    observation_variance: scalar variance of observation noise
    
    Returns:
    updated ensemble with same shape (M, N)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Ensure inputs are finite
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Local ESRF with window size
    window_size = 7  # Local window size (odd for symmetry)
    half_window = window_size // 2
    
    # Initialize result with background
    result = ensemble.copy()
    
    # For each observed index, apply local update
    for obs_idx in observed_indices:
        obs_idx = int(obs_idx)
        
        # Check time budget
        if time.time() > deadline:
            break
        
        # Define local window around this observation
        local_start = max(0, obs_idx - half_window)
        local_end = min(N, obs_idx + half_window + 1)
        
        local_indices = np.arange(local_start, local_end)
        local_len = len(local_indices)
        
        # Extract local ensemble
        local_ens = ensemble[:, local_indices].copy()
        
        # Create local observation mask
        local_obs_mask = np.zeros(local_len, dtype=bool)
        local_obs_mask[obs_idx - local_start] = True
        
        # Extract local observation
        y_local = observations[np.where(observed_indices == obs_idx)[0][0]]
        
        # Apply local ESRF update
        local_ens_updated = _local_esrf_update(local_ens, local_obs_mask, np.array([y_local]), observation_variance)
        
        # Update result with local analysis
        result[:, local_indices] = local_ens_updated
    
    # Final safety checks
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result
# EVOLVE-BLOCK-END
