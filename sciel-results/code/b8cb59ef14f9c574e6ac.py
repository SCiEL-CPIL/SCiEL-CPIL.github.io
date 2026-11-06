"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = _start_time + 145.0

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with localization and adaptive inflation
    for Lorenz-96 state reconstruction from partial observations.
    """
    if time.time() > _DEADLINE:
        return np.array(ensemble, dtype=np.float64)

    X_b = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)

    M, N = X_b.shape
    N_obs = len(obs_idx)

    # Background mean and anomalies
    mean_b = np.mean(X_b, axis=0)
    W_b = X_b - mean_b  # M x N

    # Background covariance
    P_b = (W_b.T @ W_b) / (M - 1)  # N x N

    # Extract relevant covariance blocks
    P_oo = P_b[np.ix_(obs_idx, obs_idx)]
    P_xo = P_b[:, obs_idx]  # N x N_obs

    # Observation error covariance
    R = np.eye(N_obs) * R_val

    # Innovation
    d = y - mean_b[obs_idx]

    # Localization: circular distance from all state indices to observed indices
    N_all = np.arange(N)
    dist = np.abs(N_all[:, np.newaxis] - obs_idx[np.newaxis, :])
    dist = np.minimum(dist, N - dist)

    # Gaussian taper with range L=1.5
    L = 1.5
    loc_matrix = np.exp(-0.5 * (dist / L) ** 2)  # N x N_obs

    # Localize only the cross-covariance P_xo
    P_xo_loc = P_xo * loc_matrix

    # Innovation covariance
    S = P_oo + R

    # Compute Kalman gain K = P_xo_loc @ S^{-1}
    try:
        # Add small regularization for numerical stability
        S_reg = S + 1e-10 * np.eye(N_obs)
        S_inv = np.linalg.inv(S_reg)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    K = P_xo_loc @ S_inv  # N x N_obs

    # Update mean
    mean_a = mean_b + K @ d

    # Update ensemble perturbations
    # W_a = W_b - K @ H @ W_b  (in transposed form for efficiency)
    H_W_b_T = W_b.T[obs_idx, :]  # N_obs x M
    W_a_T = W_b.T - K @ H_W_b_T  # N x M

    # Adaptive inflation
    expected_innov_var = np.mean(np.diag(S))
    actual_innov_var = np.var(d) + 1e-10

    ratio = expected_innov_var / actual_innov_var
    inflation_factor = np.sqrt(ratio)
    inflation_factor = np.clip(inflation_factor, 1.0, 1.05)

    # Apply inflation to perturbations
    W_a_T *= inflation_factor

    # Re-center to ensure mean is preserved
    mean_W_a = np.mean(W_a_T, axis=1)
    W_a_T -= mean_W_a[:, np.newaxis]

    # Construct analysis ensemble
    X_a = mean_a + W_a_T.T  # M x N

    # Safety: ensure finite and bounded
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    X_a = np.clip(X_a, -100.0, 100.0)

    return X_a
# EVOLVE-BLOCK-END
