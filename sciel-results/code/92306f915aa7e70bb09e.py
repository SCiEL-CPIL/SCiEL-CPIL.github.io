"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with localization and adaptive inflation
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
    mean_b = np.mean(X, axis=0)

    # Compute background covariance
    W = X - mean_b  # M x N anomalies
    P_b = (W.T @ W) / (M - 1)  # N x N

    # Localization: apply a correlation-based taper to P_b
    # Use a Gaussian correlation function with range parameter
    # For Lorenz-96 with periodic boundary, distance is circular
    idx = np.arange(N)
    # Compute circular distance matrix
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, N - dist)  # circular distance

    # Correlation-based localization: taper = exp(-0.5 * (dist/range)^2)
    # range ~ 2-3 for Lorenz-96
    range_param = 2.5
    taper = np.exp(-0.5 * (dist / range_param) ** 2)

    # Apply localization to background covariance
    P_loc = P_b * taper

    # Observation operator: select observed components
    P_obs = P_loc[np.ix_(obs_idx, obs_idx)] + R * np.eye(n_obs)

    # Innovation
    innovation = y - mean_b[obs_idx]

    # Kalman gain: K = P_loc[:, obs_idx] @ inv(P_obs)
    P_loc_Ht = P_loc[:, obs_idx]

    # Solve for K using Cholesky
    try:
        L = np.linalg.cholesky(P_obs)
        Z = np.linalg.solve(L.T, P_loc_Ht.T)  # n_obs x N
        K = np.linalg.solve(L, Z).T  # N x n_obs
    except np.linalg.LinAlgError:
        try:
            K = P_loc_Ht @ np.linalg.inv(P_obs)
        except np.linalg.LinAlgError:
            K = P_loc_Ht @ np.linalg.pinv(P_obs)

    # Update mean
    mean_a = mean_b + K @ innovation

    # Update ensemble members: X_a = X + K @ (y - H X)
    H_X_b = X[:, obs_idx]
    innovation_ens = y[None, :] - H_X_b  # M x n_obs
    update = innovation_ens @ K.T  # M x N
    X_a = X + update

    # Preserve the updated mean
    X_a = X_a - np.mean(X_a, axis=0) + mean_a

    # Adaptive inflation
    # Compute innovation variance
    innovation_var = np.var(innovation)
    ens_var_obs = np.var(X[:, obs_idx], axis=0)
    expected_innovation_var = np.mean(ens_var_obs + R)

    if expected_innovation_var > 0 and innovation_var > 0:
        inflation = np.sqrt(innovation_var / expected_innovation_var)
        inflation = np.clip(inflation, 1.0, 1.15)
    else:
        inflation = 1.0

    # Apply inflation
    X_a = mean_a + inflation * (X_a - mean_a)

    # Ensure finite and bounded
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    X_a = np.clip(X_a, -100.0, 100.0)

    if time.time() > DEADLINE:
        return X_a

    return X_a
# EVOLVE-BLOCK-END
