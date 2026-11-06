"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman filter with adaptive localization and inflation
    for Lorenz-96 state reconstruction from partial observations.
    """
    t_start = time.time()
    DEADLINE = 145.0

    X = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R = float(observation_variance)

    M, N = X.shape  # M ensemble members, N state dimension
    n_obs = len(obs_idx)

    # Background mean
    mean_b = X.mean(axis=0)

    # Anomalies
    W = X - mean_b  # M x N

    # Localization: Gaussian taper with circular distance
    idx = np.arange(N)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, N - dist)
    range_param = 4.0
    taper = np.exp(-0.5 * (dist / range_param) ** 2)
    # Ensure diagonal is exactly 1.0
    np.fill_diagonal(taper, 1.0)

    # Localized background covariance
    P_b = (W.T @ W) / (M - 1)
    P_loc = P_b * taper

    # Observation-space covariance
    H_idx = obs_idx
    P_obs = P_loc[np.ix_(H_idx, H_idx)] + R * np.eye(n_obs)
    # Ensure positive definiteness with small jitter
    P_obs += 1e-10 * np.eye(n_obs)

    # Innovation
    innovation = y - mean_b[H_idx]

    # Compute Kalman gain via Cholesky
    try:
        L = np.linalg.cholesky(P_obs)
        # K = P_loc[:, H_idx] @ P_obs^{-1}
        # Using Cholesky: K = (P_loc[:, H_idx] @ L^{-T}) @ L^{-1}
        temp = np.linalg.solve(L, P_loc[:, H_idx].T)  # n_obs x N
        K = np.linalg.solve(L.T, temp).T  # N x n_obs
    except np.linalg.LinAlgError:
        # Fallback: use SVD for pseudo-inverse
        try:
            U, s, Vt = np.linalg.svd(P_obs, full_matrices=False)
            s_inv = np.where(s > 1e-12, 1.0 / s, 0.0)
            P_obs_inv = (Vt.T * s_inv) @ U.T
            K = P_loc[:, H_idx] @ P_obs_inv
        except Exception:
            # Last resort: simple diagonal approximation
            diag_P_obs = np.diag(P_obs)
            K = P_loc[:, H_idx] * (1.0 / np.maximum(diag_P_obs, 1e-12))[None, :]

    # Updated mean
    mean_a = mean_b + K @ innovation

    # Standard ensemble Kalman filter update:
    # X_a_i = X_b_i + K @ (y - H X_b_i)
    # Centered form: X_a = mean_a + W + K @ (H W - innovation)
    HW = W[:, H_idx]  # M x n_obs
    innovation_members = innovation[None, :] - HW  # M x n_obs

    # Update
    X_a = X + (innovation_members @ K.T)  # M x N

    # Re-center to exact mean
    X_a = X_a - X_a.mean(axis=0) + mean_a

    # Adaptive inflation: compare ensemble spread in observation space to innovation
    ens_obs_var = np.var(HW, axis=0)  # per-observed-variable ensemble variance
    expected_innov_var = ens_obs_var + R  # expected innovation variance
    actual_innov_var = np.var(innovation)

    if expected_innov_var.mean() > 1e-10 and actual_innov_var > 1e-10:
        # Ratio of actual to expected innovation variance
        ratio = actual_innov_var / expected_innov_var.mean()
        # Inflation factor: if ensemble is too tight (ratio > 1), inflate
        # Use a softer, more responsive formula
        inflation_factor = np.sqrt(np.clip(ratio, 0.5, 2.0))
        # Allow more aggressive inflation for chaotic systems
        inflation_factor = np.clip(inflation_factor, 0.95, 1.3)
    else:
        inflation_factor = 1.0

    # Apply inflation around the updated mean
    X_a = mean_a + inflation_factor * (X_a - mean_a)

    # Ensure finite and bounded
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    X_a = np.clip(X_a, -100.0, 100.0)

    if time.time() > DEADLINE:
        return X_a

    return X_a
# EVOLVE-BLOCK-END
