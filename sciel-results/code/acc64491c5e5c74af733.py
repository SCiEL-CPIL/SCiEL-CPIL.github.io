"""Combined ensemble assimilation for Lorenz-96 partial observation."""
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
    Ensemble Kalman Filter with:
    - Gaussian localization of cross-covariance (tuned for L96)
    - Deterministic EnSRF perturbation update
    - Adaptive multiplicative inflation based on innovation consistency
    - Deadline guard
    """
    M, N = ensemble.shape
    obs_var = float(observation_variance)
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
    W_b = X_b - mean_b  # (M, N)

    # Background covariance (N x N)
    P_b = (W_b.T @ W_b) / (M - 1)

    # Observation operator components
    P_oo = P_b[np.ix_(obs_idx, obs_idx)]
    P_xo = P_b[:, obs_idx]  # (N, L)

    # Innovation covariance
    S = P_oo + obs_var * np.eye(L)

    # Gaussian localization of cross-covariance
    # Periodic distance on ring of length N
    state_idx = np.arange(N)
    diffs = np.abs(state_idx[:, None] - obs_idx[None, :])
    distances = np.minimum(diffs, N - diffs)  # (N, L)

    # Tuned correlation length scale for L96 with N=20
    L_loc = 4.0
    loc_matrix = np.exp(-0.5 * (distances / L_loc) ** 2)  # (N, L)

    P_xo_loc = P_xo * loc_matrix  # (N, L)

    # Also localize the observation covariance for consistency
    obs_diffs = np.abs(obs_idx[:, None] - obs_idx[None, :])
    obs_distances = np.minimum(obs_diffs, N - obs_diffs)
    obs_loc_matrix = np.exp(-0.5 * (obs_distances / L_loc) ** 2)
    P_oo_loc = P_oo * obs_loc_matrix

    S_loc = P_oo_loc + obs_var * np.eye(L)

    # Kalman gain: K = P_xo_loc @ S^{-1}
    try:
        S_inv_PbH = np.linalg.solve(S_loc.T, P_xo_loc.T)  # (L, N)
        K = S_inv_PbH.T  # (N, L)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S_loc)
        K = P_xo_loc @ S_inv  # (N, L)

    # Mean update
    d = y_obs[0] - mean_b[obs_idx]  # innovation vector (L,)
    mean_a = mean_b + K @ d  # (N,)

    # Perturbation update (deterministic EnSRF style)
    H_W_b_T = W_b.T[obs_idx, :]  # (L, M)
    K_HW = K @ H_W_b_T  # (N, M)
    W_a_T = W_b.T - K_HW  # (N, M)

    # Adaptive multiplicative inflation based on innovation consistency
    # Compare actual innovation variance to expected
    expected_innov_var = np.mean(np.diag(S_loc))
    actual_innov_var = np.var(d) + 1e-10
    ratio = actual_innov_var / expected_innov_var
    inflation_factor = np.sqrt(ratio)
    inflation_factor = np.clip(inflation_factor, 1.0, 1.12)

    # Apply inflation to perturbations
    W_a_T *= inflation_factor

    # Re-center perturbations
    mean_W_a = np.mean(W_a_T, axis=1)
    W_a_T -= mean_W_a[:, np.newaxis]

    # Reconstruct ensemble
    X_a = mean_a[None, :] + W_a_T.T  # (M, N)

    # Safety: ensure finite and clip
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    np.clip(X_a, -100.0, 100.0, out=X_a)

    return X_a

# EVOLVE-BLOCK-END
