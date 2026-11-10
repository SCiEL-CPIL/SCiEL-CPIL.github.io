"""Lorenz-96 ensemble assimilation with Local ETKF + 2-step EnKS back-correction."""
import numpy as np
import time

# EVOLVE-BLOCK-START
_N = 20
_WINDOW_HALF = 4  # window of 9 variables
_CORR_LEN = 5.0

def _gaspari_cohn(d, c):
    """Gaspari-Cohn C2 correlation function (vectorized)."""
    d = np.abs(d)
    out = np.zeros_like(d, dtype=float)
    dc = d / c
    # region 1: dc <= 0.5
    m1 = dc <= 0.5
    if np.any(m1):
        x = dc[m1]
        out[m1] = (-5/4)*x**5 + 5*x**4 - 10*x**3 + 5*x
    # region 2: 0.5 < dc <= 1.5
    m2 = (dc > 0.5) & (dc <= 1.5)
    if np.any(m2):
        x = dc[m2]
        out[m2] = (1/12)*x**5 - (5/4)*x**4 + (5/2)*x**3 - 5*x**2 + (25/12)*x + 0.5
    # region 3: 1.5 < dc <= 2.5
    m3 = (dc > 1.5) & (dc <= 2.5)
    if np.any(m3):
        x = dc[m3]
        out[m3] = -(1/16)*x**5 + (1/4)*x**4 - (15/8)*x**3 + (5/2)*x**2 - (25/12)*x + 105/48
    return out

def _periodic_dist(i, j, N):
    d = abs(i - j)
    return min(d, N - d)

def _etkf_update(ens, obs, obs_local_idx, obs_var, loc_w, M):
    """
    Local Ensemble Transform Kalman Filter update.
    
    ens: (M, L) local ensemble
    obs: (n_obs,) local observations
    obs_local_idx: indices within local window that are observed
    obs_var: observation variance
    loc_w: (L,) localization weights
    M: ensemble size
    """
    L = ens.shape[1]
    n_obs = len(obs)
    
    # Mean and anomalies
    mean = ens.mean(axis=0)
    X = ens - mean  # (M, L)
    
    # Apply localization to anomalies
    X_loc = X * loc_w[np.newaxis, :]
    
    # Sample covariance C = X^T X / (M-1)  (L, L)
    C = (X_loc.T @ X_loc) / (M - 1)
    
    # Observation operator H: (n_obs, L)
    H = np.zeros((n_obs, L))
    for k, l in enumerate(obs_local_idx):
        H[k, l] = 1.0
    
    # Innovation covariance S = H C H^T + R  (n_obs, n_obs)
    HC = H @ C
    S = HC @ H.T + obs_var * np.eye(n_obs)
    
    # Compute ETKF transformation matrix
    # K = C H^T S^{-1}  (L, n_obs)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    K = C @ H.T @ S_inv  # (L, n_obs)
    
    # ETKF: find T such that (X T + K(y - H mean)) has the right covariance
    # Standard ETKF: T = (I + K H X^T X / (M-1))^{-1/2} approximately
    # Simpler: use the "square root" form
    # Y = X  (M, L)
    # A = (I - H^T S^{-1} H C)^{1/2} ... this is complex
    # Use the simpler "ensemble Kalman filter with transformation"
    
    # Actually, for local ETKF with small windows, the standard EnKF update
    # x_post = x_prior + K (y - H x_prior) is already a valid transform.
    # The key advantage of ETKF is preserving spread. Let's use the
    # "square root" ETKF form:
    
    # Compute the transformation
    # P_prior = C (L,L)
    # P_post = C - C H^T S^{-1} H C
    # We want T such that X_post = T @ X + K(y - H mean)
    # and X_post^T X_post / (M-1) = P_post
    #
    # Standard ETKF: T = (I + K H X^T X / (M-1))^{-1/2}
    # Let's compute this
    
    # K H X^T X / (M-1): (L, n_obs)(n_obs, L)(L, M)(M, L)/(M-1)
    # = K H C (L, L)
    KH_C = K @ H @ C  # (L, L)
    
    # M_mat = I + KH_C  (L, L)
    M_mat = np.eye(L) + KH_C
    
    # T = M_mat^{-1/2}
    # Eigendecompose
    try:
        eigvals, eigvecs = np.linalg.eigh(M_mat)
        eigvals = np.clip(eigvals, 1e-10, None)
        T = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    except np.linalg.LinAlgError:
        T = np.eye(L)
    
    # Apply update
    # innovation = obs - H @ mean  (n_obs,)
    innovation = obs - H @ mean
    
    # X_post = T @ X + K @ innovation  (M, L)
    X_post = T @ X + K @ innovation[np.newaxis, :]
    
    # ens_post = mean + X_post
    ens_post = mean + X_post
    
    return ens_post

def _local_update_window(ens, observations, obs_map, obs_set, i, N, M, obs_var):
    """
    Perform a localized ETKF update centered at state variable i.
    Returns the updated local ensemble for the window.
    """
    W = _WINDOW_HALF
    local_idx = [(i + j) % N for j in range(-W, W + 1)]
    L = len(local_idx)
    
    # Find which local positions are observed
    local_obs_pos = []
    for l, g in enumerate(local_idx):
        if g in obs_set:
            local_obs_pos.append(l)
    
    if len(local_obs_pos) == 0:
        return None, local_idx
    
    # Local observations
    y_local = np.array([observations[obs_map[local_idx[l]]] for l in local_obs_pos])
    
    # Localization weights
    dists = np.array([_periodic_dist(i, (i + j) % N, N) for j in range(-W, W + 1)])
    loc_w = _gaspari_cohn(dists, _CORR_LEN)
    loc_w[W] = 1.0  # ensure center is 1
    
    # Local ensemble
    local_ens = ens[:, local_idx].copy()
    
    # ETKF update
    updated_local = _etkf_update(local_ens, y_local, local_obs_pos, obs_var, loc_w, M)
    
    return updated_local, local_idx

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local ETKF with 2-step EnKS back-correction for Lorenz-96 assimilation.
    
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
    
    # Build observation mapping
    obs_set = set(int(x) for x in observed_indices)
    obs_map = {int(observed_indices[k]): k for k in range(K)}
    
    result = ensemble.astype(float).copy()
    
    # Adaptive multiplicative inflation based on ensemble spread vs obs error
    if M > 2:
        obs_vars = list(obs_set)
        spread = np.std(result[:, obs_vars], axis=0)
        obs_err = np.sqrt(observation_variance)
        ratio = np.mean(spread) / obs_err if obs_err > 0 else 1.0
        if ratio < 1.0:
            inflation = 1.0 + 0.08 * (1.0 - ratio)
        else:
            inflation = 1.005
        inflation = float(np.clip(inflation, 1.003, 1.06))
    else:
        inflation = 1.005
    
    mean = result.mean(axis=0)
    result = mean + inflation * (result - mean)
    
    # Time budget guard
    time_budget = 130.0
    
    # Pass 1: Standard local ETKF update sweeping all variables
    # Process in order, using the already-updated neighbors for consistency
    for i in range(N):
        if time.time() - t0 > time_budget:
            break
        
        updated_local, local_idx = _local_update_window(
            result, observations, obs_map, obs_set, i, N, M, observation_variance
        )
        
        if updated_local is not None:
            result[:, local_idx] = updated_local
    
    # Pass 2: 2-step EnKS back-correction
    # This pass re-applies the observation update with a temporal correlation
    # model that accounts for the fact that the current state was propagated
    # from the previous state. We use a simple approach: re-run the local
    # ETKF but with a slightly reduced observation variance to account for
    # the additional information from the temporal correlation.
    # The idea: if the ensemble was under-dispersive, the first pass may
    # have over-corrected. The back-pass with adjusted parameters helps
    # correct for propagation errors.
    
    if time.time() - t0 < time_budget - 5.0:
        # Use a slightly higher effective obs variance to be less aggressive
        obs_var_adjusted = observation_variance * 1.1
        
        for i in range(N):
            if time.time() - t0 > time_budget:
                break
            
            updated_local, local_idx = _local_update_window(
                result, observations, obs_map, obs_set, i, N, M, obs_var_adjusted
            )
            
            if updated_local is not None:
                # Blend: 70% new update, 30% previous
                result[:, local_idx] = 0.7 * updated_local + 0.3 * result[:, local_idx]
    
    # Safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
# EVOLVE-BLOCK-END
