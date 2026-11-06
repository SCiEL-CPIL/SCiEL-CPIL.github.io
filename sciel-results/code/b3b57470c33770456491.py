"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_start_time = None

def _get_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Block-Localized Ensemble Kalman Filter with:
    - Block localization (radius 4, overlap 4)
    - Adaptive multiplicative inflation per block
    - Deadline guard
    """
    M, N = ensemble.shape
    obs_var = observation_variance
    obs_idx = np.asarray(observed_indices, dtype=int)
    
    # Deadline guard
    elapsed = _get_time()
    if elapsed > 140.0:
        result = np.array(ensemble, dtype=np.float64).copy()
        obs_arr = np.asarray(observations, dtype=np.float64)
        if obs_arr.ndim == 1:
            obs_arr = np.tile(obs_arr, (M, 1))
        result[:, obs_idx] += 0.5 * (obs_arr - result[:, obs_idx])
        np.clip(result, -100.0, 100.0, out=result)
        return result
    
    X_b = np.array(ensemble, dtype=np.float64)
    y_obs = np.asarray(observations, dtype=np.float64)
    if y_obs.ndim == 1:
        y_obs = np.tile(y_obs, (M, 1))
    
    L_obs = len(obs_idx)
    
    # Block configuration
    # N=20, Block size B=8, Overlap O=4, Step S=4
    B = 8
    S = 4
    num_blocks = N // S  # 5 blocks
    
    # Initialize output ensemble
    X_a = np.zeros_like(X_b)
    # Count how many blocks cover each variable for averaging
    counts = np.zeros(N)
    
    for k in range(num_blocks):
        start = k * S
        # Indices in this block (mod N)
        block_idx = (np.arange(start, start + B)) % N
        
        # Extract local ensemble
        X_local = X_b[:, block_idx]  # (M, B)
        
        # Determine which observations fall within this block
        # We need to map global obs_idx to local indices
        # obs_idx is global. block_idx is global.
        # Find intersection
        local_obs_mask = np.isin(block_idx, obs_idx)
        local_obs_global_idx = block_idx[local_obs_mask]
        
        if len(local_obs_global_idx) == 0:
            # No observations in this block, keep background
            X_a[:, block_idx] += X_local
            counts[block_idx] += 1
            continue
        
        # Extract local observations
        # y_obs is (M, L_obs). We need the columns corresponding to local_obs_global_idx
        # Find the indices in obs_idx that match local_obs_global_idx
        obs_positions = []
        for goi in local_obs_global_idx:
            # Find index in obs_idx
            pos = np.where(obs_idx == goi)[0]
            if len(pos) > 0:
                obs_positions.append(pos[0])
        
        y_local = y_obs[:, obs_positions]  # (M, L_local)
        L_local = len(local_obs_global_idx)
        
        # Local Background Statistics
        mean_b_local = np.mean(X_local, axis=0)  # (B,)
        X_anom_local = X_local - mean_b_local  # (M, B)
        
        # Local Background Covariance (B x B)
        P_b_local = (X_anom_local.T @ X_anom_local) / (M - 1)
        
        # Local Observation Operator
        # H maps B variables to L_local observations
        # It's a selection matrix
        H_local = np.zeros((L_local, B))
        for i, goi in enumerate(local_obs_global_idx):
            # Find index of goi in block_idx
            local_pos = np.where(block_idx == goi)[0][0]
            H_local[i, local_pos] = 1.0
        
        # Innovation Covariance S (L_local x L_local)
        P_byy = H_local @ P_b_local @ H_local.T
        S = P_byy + obs_var * np.eye(L_local)
        
        # Cross Covariance P_bH (B x L_local)
        P_bH = P_b_local @ H_local.T
        
        # Kalman Gain K (B x L_local)
        try:
            S_inv_PbH = np.linalg.solve(S.T, P_bH.T)  # (L_local, B)
            K = S_inv_PbH.T  # (B, L_local)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            K = P_bH @ S_inv  # (B, L_local)
        
        # Analysis Update
        # Member innovations
        X_obs_members = X_local @ H_local.T  # (M, L_local)
        member_innovations = y_local - X_obs_members  # (M, L_local)
        
        # X_a_local = X_b_local + member_innovations @ K.T
        X_a_local = X_local + member_innovations @ K.T  # (M, B)
        
        # Adaptive Multiplicative Inflation (Local)
        spread_b = np.std(X_local, axis=0, ddof=1)  # (B,)
        spread_a = np.std(X_a_local, axis=0, ddof=1)  # (B,)
        
        min_ratio = 0.85
        inflation_factors = np.ones(B)
        for i in range(B):
            if spread_b[i] > 1e-6:
                ratio = spread_a[i] / spread_b[i]
                if ratio < min_ratio:
                    inflation_factors[i] = min(1.20, min_ratio / ratio)
        
        if np.any(inflation_factors > 1.0):
            mean_a_local = np.mean(X_a_local, axis=0)
            X_a_local = mean_a_local[None, :] + inflation_factors[None, :] * (X_a_local - mean_a_local[None, :])
        
        # Accumulate for averaging in overlaps
        X_a[:, block_idx] += X_a_local
        counts[block_idx] += 1
    
    # Average overlapping regions
    # Avoid division by zero (should not happen with this config)
    X_a /= counts[None, :]
    
    # Safety: ensure finite and clip
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    np.clip(X_a, -100.0, 100.0, out=X_a)
    
    return X_a

# EVOLVE-BLOCK-END
