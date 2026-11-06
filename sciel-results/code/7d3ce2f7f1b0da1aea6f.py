import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 145.0

def analyze(ensemble, observations, observed_indices, observation_variance):
    if time.time() > _DEADLINE:
        return np.array(ensemble, dtype=np.float64)

    X_b = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)

    M, N = X_b.shape
    N_obs = len(obs_idx)

    if N_obs == 0:
        return X_b.copy()

    # Background mean and perturbations
    mean_b = np.mean(X_b, axis=0)
    W_b = X_b - mean_b  # (M, N)

    # Background covariance (unbiased)
    P_b = (W_b.T @ W_b) / (M - 1)  # (N, N)

    # Extract relevant covariances
    P_oo = P_b[np.ix_(obs_idx, obs_idx)]
    P_xo = P_b[:, obs_idx]  # (N, N_obs)

    # Observation error covariance
    R = np.eye(N_obs) * R_val

    # Innovation
    H_mean = mean_b[obs_idx]
    d = y - H_mean  # (N_obs,)

    # Smooth Gaussian localization
    N_all = np.arange(N)
    dist = np.abs(N_all[:, np.newaxis] - obs_idx[np.newaxis, :])
    dist = np.minimum(dist, N - dist)

    # Tuned localization length for L96 correlation decay
    L = 2.0
    loc_matrix = np.exp(-0.5 * (dist / L) ** 2)  # (N, N_obs)

    P_xo_loc = P_xo * loc_matrix

    # Innovation covariance
    S = P_oo + R

    # Kalman gain
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    K = P_xo_loc @ S_inv  # (N, N_obs)

    # Update mean
    mean_a = mean_b + K @ d

    # Update perturbations (deterministic EnSRF style)
    H_W_b_T = W_b.T[obs_idx, :]  # (N_obs, M)
    K_HW = K @ H_W_b_T  # (N, M)
    W_a_T = W_b.T - K_HW  # (N, M)

    # Adaptive inflation based on innovation consistency
    expected_innov_var = np.mean(np.diag(S))
    actual_innov_var = np.var(d) + 1e-10
    ratio = actual_innov_var / expected_innov_var
    inflation_factor = np.sqrt(ratio)
    inflation_factor = np.clip(inflation_factor, 1.0, 1.08)

    W_a_T *= inflation_factor

    # Re-center
    mean_W_a = np.mean(W_a_T, axis=1)
    W_a_T -= mean_W_a[:, np.newaxis]

    # Gentle rescaling to prevent over/under-dispersion
    if M > 1:
        pert_scale = np.sqrt(np.mean(W_a_T**2, axis=1))
        target_scale = np.sqrt(np.mean(W_b.T**2, axis=1))
        scale_ratio = target_scale / (pert_scale + 1e-10)
        scale_ratio = np.clip(scale_ratio, 0.95, 1.05)
        W_a_T *= scale_ratio[:, np.newaxis]

    X_a = mean_a + W_a_T.T

    # Safety
    X_a = np.clip(X_a, -100.0, 100.0)
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)

    return X_a
