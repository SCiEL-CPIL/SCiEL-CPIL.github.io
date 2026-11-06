"""Combined ensemble assimilation for Lorenz-96 partial observation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Deterministic Ensemble Square-Root Kalman Filter (EnSRF) with:
    - Gaussian localization of cross-covariance (tuned for L96)
    - Adaptive multiplicative inflation based on innovation consistency
    - Gentle perturbation rescaling to prevent over/under-dispersion
    - Deadline guard
    """
    M, N = ensemble.shape
    obs_var = float(observation_variance)
    obs_idx = np.asarray(observed_indices, dtype=int)

    # Deadline guard: return background with light observation pull if over time
    if _elapsed() > 140.0:
        result = np.array(ensemble, dtype=np.float64).copy()
        y_obs = np.asarray(observations, dtype=np.float64)
        if y_obs.ndim == 1:
            y_obs = np.tile(y_obs, (M, 1))
        result[:, obs_idx] += 0.5 * (y_obs - result[:, obs_idx])
        np.clip(result, -100.0, 100.0, out=result)
        return result

    X_b = np.array(ensemble, dtype=np.float64)
    y_obs = np.asarray(observations, dtype=np.float64)
    if y_obs.ndim == 1:
        y_obs = np.tile(y_obs, (M, 1))

    L = len(obs_idx)
    if L == 0:
        return X_b.copy()

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

    L_loc = 2.0  # correlation length scale tuned for L96
    loc_matrix = np.exp(-0.5 * (distances / L_loc) ** 2)  # (N, L)

    P_xo_loc = P_xo * loc_matrix  # (N, L)

    # Kalman gain: K = P_xo_loc @ S^{-1}
    try:
        # Solve S @ X = P_xo_loc.T  =>  X = S^{-1} @ P_xo_loc.T
        # Then K = P_xo_loc @ S^{-1} = (S^{-1} @ P_xo_loc.T).T
        # But we need K = P_xo_loc @ S^{-1}, shape (N, L)
        # Let's compute S_inv = inv(S), then K = P_xo_loc @ S_inv
        S_inv = np.linalg.inv(S)
        K = P_xo_loc @ S_inv  # (N, L)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K = P_xo_loc @ S_inv

    # Innovation
    d = y_obs[0, obs_idx] - mean_b[obs_idx]  # (L,)

    # Update mean
    mean_a = mean_b + K @ d

    # Update perturbations (deterministic EnSRF)
    # W_a = W_b - K @ (H W_b - d) ... but centered form:
    # For each member i: W_a_i = W_b_i - K @ (H W_b_i)
    # Because H W_b_i is the observation-space anomaly for member i
    H_W_b = W_b[:, obs_idx]  # (M, L)
    # K @ H_W_b_i for each member
    K_HW = H_W_b @ K.T  # (M, N)
    W_a = W_b - K_HW  # (M, N)

    # Adaptive inflation based on innovation consistency
    expected_innov_var = np.mean(np.diag(S))
    actual_innov_var = np.var(d) + 1e-10
    ratio = actual_innov_var / expected_innov_var
    inflation_factor = np.sqrt(ratio)
    inflation_factor = np.clip(inflation_factor, 1.0, 1.08)

    W_a *= inflation_factor

    # Re-center perturbations to zero mean
    W_a -= W_a.mean(axis=0)

    # Gentle rescaling to prevent over/under-dispersion
    if M > 1:
        pert_scale = np.sqrt(np.mean(W_a**2, axis=0))  # (N,)
        target_scale = np.sqrt(np.mean(W_b**2, axis=0))  # (N,)
        scale_ratio = target_scale / (pert_scale + 1e-10)
        scale_ratio = np.clip(scale_ratio, 0.95, 1.05)
        W_a *= scale_ratio[None, :]

    # Reconstruct ensemble
    X_a = mean_a[None, :] + W_a

    # Safety
    X_a = np.clip(X_a, -100.0, 100.0)
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)

    return X_a

# EVOLVE-BLOCK-END
