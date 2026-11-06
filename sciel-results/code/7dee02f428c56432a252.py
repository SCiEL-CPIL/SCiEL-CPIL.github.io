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
    Local Ensemble Transform Kalman Filter (LETKF) with:
    - Block-localization radius 4 (periodic ring)
    - Analytic spread preservation via transform coefficients
    - Adaptive multiplicative inflation with temporal memory
    - Deadline guard
    """
    M, N = ensemble.shape
    obs_var = float(observation_variance)
    obs_idx = np.asarray(observed_indices, dtype=int)
    L = len(obs_idx)

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

    # Background statistics
    mean_b = np.mean(X_b, axis=0)
    X_anom = X_b - mean_b  # (M, N)

    # Block-localization radius
    radius = 4

    # For each state variable, find which observations are within the localization block
    # and which state variables are within the block (for the local analysis)
    # We'll process each state variable independently using its local block

    X_a = np.zeros_like(X_b)

    state_idx = np.arange(N)

    for i in range(N):
        # Determine local state variables: those within radius of i (periodic)
        diffs_state = np.abs(state_idx - i)
        diffs_state = np.minimum(diffs_state, N - diffs_state)
        local_state_mask = diffs_state <= radius
        local_state_idx = np.where(local_state_mask)[0]

        # Determine local observations: those within radius of i (periodic)
        diffs_obs = np.abs(obs_idx - i)
        diffs_obs = np.minimum(diffs_obs, N - diffs_obs)
        local_obs_mask = diffs_obs <= radius
        local_obs_idx = obs_idx[local_obs_mask]
        local_obs_pos = np.where(local_obs_mask)[0]  # positions within obs_idx array

        n_local_state = len(local_state_idx)
        n_local_obs = len(local_obs_idx)

        if n_local_obs == 0:
            # No observations in block; keep background
            X_a[:, i] = X_b[:, i]
            continue

        # Extract local anomalies
        local_anom_state = X_anom[:, local_state_idx]  # (M, n_local_state)
        local_anom_obs = X_anom[:, local_obs_idx]  # (M, n_local_obs)

        # Local background covariance: P_b (n_local_state x n_local_state)
        P_b_local = (local_anom_state.T @ local_anom_state) / (M - 1)

        # Cross-covariance: P_bH (n_local_state x n_local_obs)
        P_bH_local = (local_anom_state.T @ local_anom_obs) / (M - 1)

        # Observation covariance block
        P_oo = (local_anom_obs.T @ local_anom_obs) / (M - 1)
        S = P_oo + obs_var * np.eye(n_local_obs)

        # LETKF transform: compute T such that (X_anom @ T) has the analysis covariance
        # The transform T is upper triangular, computed via Cholesky-like factorization
        # T = (I - K H) where K is the Kalman gain, but we compute it directly

        # Compute the innovation covariance inverse
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        # Kalman gain for the local block: K = P_bH @ S_inv (n_local_state x n_local_obs)
        K_local = P_bH_local @ S_inv

        # The analysis anomalies are: X_anom_a = X_anom_b @ T
        # where T = I - K @ H_local, and H_local maps state to obs within the block
        # H_local is (n_local_state x n_local_obs) with 1s where state var is observed

        # Build H_local: for each local state variable, which local obs does it correspond to?
        H_local = np.zeros((n_local_state, n_local_obs))
        for li, si in enumerate(local_state_idx):
            for lj, oj in enumerate(local_obs_idx):
                if si == oj:
                    H_local[li, lj] = 1.0

        # T = I - K @ H_local (n_local_state x n_local_state)
        T = np.eye(n_local_state) - K_local @ H_local

        # Make T upper triangular via Cholesky-like decomposition for stability
        # Actually, the standard LETKF uses a specific factorization.
        # Let's use the approach: factor T = L @ U where L is lower triangular
        # and use U^T for the transform to preserve spread
        try:
            # Cholesky of T (should be SPD if inflation is adequate)
            L_chol = np.linalg.cholesky(T)
            # Transform: X_anom_a = X_anom_b @ L_chol
            local_anom_a = local_anom_state @ L_chol
        except np.linalg.LinAlgError:
            # Fallback: use SVD-based approach
            U_svd, s_vals, Vt_svd = np.linalg.svd(T)
            # Use the square root: T^{1/2}
            T_sqrt = U_svd @ np.diag(np.sqrt(s_vals)) @ Vt_svd
            local_anom_a = local_anom_state @ T_sqrt

        # Reconstruct local analysis
        local_mean_a = mean_b[local_state_idx] + local_anom_a.mean(axis=0)
        X_a[:, local_state_idx] = local_mean_a[None, :] + local_anom_a

    # Adaptive multiplicative inflation (per-variable, with memory)
    spread_b = np.std(X_b, axis=0, ddof=1)
    spread_a = np.std(X_a, axis=0, ddof=1)

    if N not in _inflation_state:
        _inflation_state[N] = np.ones(N)
    prev_inflation = _inflation_state[N]

    min_ratio = 0.85
    inflation_factors = np.ones(N)
    for i in range(N):
        if spread_b[i] > 1e-6:
            ratio = spread_a[i] / spread_b[i]
            if ratio < min_ratio:
                target = min(1.25, min_ratio / max(ratio, 1e-6))
                inflation_factors[i] = 0.5 * prev_inflation[i] + 0.5 * target
            else:
                inflation_factors[i] = 0.7 * prev_inflation[i] + 0.3 * 1.0
        else:
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
