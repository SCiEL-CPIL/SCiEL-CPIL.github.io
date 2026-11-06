"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_start_time = None
_inflation_state = {}

def _get_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with:
    - Localized cross-covariance (exponential decay)
    - Adaptive multiplicative inflation (per-variable, with memory)
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
    
    L = len(obs_idx)
    
    # Background statistics
    mean_b = np.mean(X_b, axis=0)
    X_anom = X_b - mean_b  # (M, N)
    
    # Background covariance (N x N)
    P_b = (X_anom.T @ X_anom) / (M - 1)
    
    # Observation operator components
    P_byy = P_b[np.ix_(obs_idx, obs_idx)]
    P_bH = P_b[:, obs_idx]  # (N, L)
    
    # Innovation covariance
    S = P_byy + obs_var * np.eye(L)
    
    # Exponential localization of cross-covariance
    # Periodic distance on ring of length N
    state_idx = np.arange(N)
    diffs = np.abs(state_idx[:, None] - obs_idx[None, :])
    distances = np.minimum(diffs, N - diffs)  # (N, L)
    
    L_loc = 3.0  # correlation length scale
    loc_matrix = np.exp(-distances / L_loc)  # (N, L)
    
    P_bH_loc = P_bH * loc_matrix  # (N, L)
    
    # Kalman gain: K = P_bH_loc @ S^{-1}
    try:
        S_inv_PbH = np.linalg.solve(S.T, P_bH_loc.T)  # (L, N)
        K = S_inv_PbH.T  # (N, L)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K = P_bH_loc @ S_inv  # (N, L)
    
    # Analysis update
    # Member innovations: y_obs - H(X_b) for each member
    X_obs_members = X_b[:, obs_idx]  # (M, L)
    member_innovations = y_obs - X_obs_members  # (M, L)
    
    # X_a = X_b + member_innovations @ K.T
    X_a = X_b + member_innovations @ K.T  # (M, N)
    
    # Adaptive multiplicative inflation (per-variable, with memory)
    spread_b = np.std(X_b, axis=0, ddof=1)  # (N,)
    spread_a = np.std(X_a, axis=0, ddof=1)  # (N,)
    
    # Use stored inflation state for smoother adaptation
    if N not in _inflation_state:
        _inflation_state[N] = np.ones(N)
    
    prev_inflation = _inflation_state[N]
    
    min_ratio = 0.80
    inflation_factors = np.ones(N)
    for i in range(N):
        if spread_b[i] > 1e-6:
            ratio = spread_a[i] / spread_b[i]
            if ratio < min_ratio:
                target = min(1.30, min_ratio / ratio)
                # Smooth transition: blend previous and new inflation
                inflation_factors[i] = 0.5 * prev_inflation[i] + 0.5 * target
            else:
                # Decay back toward 1.0
                inflation_factors[i] = 0.7 * prev_inflation[i] + 0.3 * 1.0
    
    _inflation_state[N] = inflation_factors.copy()
    
    if np.any(inflation_factors > 1.0):
        mean_a = np.mean(X_a, axis=0)
        X_a = mean_a[None, :] + inflation_factors[None, :] * (X_a - mean_a[None, :])
    
    # Safety: ensure finite and clip
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    np.clip(X_a, -100.0, 100.0, out=X_a)
    
    return X_a

# EVOLVE-BLOCK-END
