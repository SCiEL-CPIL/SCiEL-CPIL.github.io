"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble square-root Kalman filter with localization and adaptive inflation
    for Lorenz-96 state reconstruction from partial observations.
    """
    t_start = time.time()
    DEADLINE = 148.0

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
    # Ensure diagonal is 1.0
    np.fill_diagonal(taper, 1.0)

    # Localized background covariance
    P_b = (W.T @ W) / (M - 1)
    P_loc = P_b * taper

    # Observation-space covariance
    H_idx = obs_idx
    P_obs = P_loc[np.ix_(H_idx, H_idx)] + R * np.eye(n_obs)
    # Ensure positive definiteness
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
        except np.linalg.LinAlgError:
            K = np.zeros((N, n_obs))

    # Updated mean
    mean_a = mean_b + K @ innovation

    # Standard EnKF update: X_a_i = X_b_i + K @ (y - H X_b_i)
    # Using centered form: X_a_i = mean_a + W_i + K @ (HW_i - innovation)
    HW = W[:, H_idx]  # M x n_obs
    
    # Update anomalies
    W_a = W + (HW - innovation[None, :]) @ K.T  # M x N
    
    # Updated ensemble
    X_a = mean_a + W_a

    # Adaptive inflation based on innovation statistics
    # Compare innovation variance to expected, per observation
    ens_obs_var = W[:, H_idx].var(axis=0)  # n_obs
    expected_innov_var = ens_obs_var + R  # n_obs
    actual_innov_var = innovation ** 2  # n_obs
    
    # Use median to be robust to outliers
    med_expected = np.median(expected_innov_var)
    med_actual = np.median(actual_innov_var)
    
    if med_expected > 1e-10 and med_actual > 1e-10:
        inflation_factor = np.sqrt(med_actual / med_expected)
        inflation_factor = np.clip(inflation_factor, 1.0, 1.15)
    else:
        inflation_factor = 1.0

    X_a = mean_a + inflation_factor * (X_a - mean_a)

    # Spatial regularization for unobserved variables:
    # Apply a mild smoothing to unobserved coordinates to enforce
    # physical consistency (Lorenz-96 has smooth spatial structure)
    unobs_idx = np.setdiff1d(np.arange(N), H_idx)
    if len(unobs_idx) > 0:
        # Compute spatial smoothing weights using the same taper
        # Blend each unobserved variable with its neighbors
        alpha = 0.1  # Mild smoothing strength
        for i in unobs_idx:
            # Get neighbors with weights
            neighbor_weights = taper[i, :]
            neighbor_weights[i] = 0.0  # Exclude self
            w_sum = neighbor_weights.sum()
            if w_sum > 1e-10:
                neighbor_mean = (X_a[:, unobs_idx] * neighbor_weights[unobs_idx]).sum(axis=1) / w_sum
                X_a[:, i] = (1 - alpha) * X_a[:, i] + alpha * neighbor_mean

    # Ensure finite and bounded
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    X_a = np.clip(X_a, -100.0, 100.0)

    if time.time() > DEADLINE:
        return X_a

    return X_a
# EVOLVE-BLOCK-END
