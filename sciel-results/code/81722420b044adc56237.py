"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Localized Ensemble Kalman Filter (EnKF) with adaptive inflation
    for Lorenz-96 state reconstruction from partial observations.
    Uses correlation-based localization with a tuned range parameter
    and robust Kalman gain computation.
    """
    t_start = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill

    X = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R = float(observation_variance)

    M, N = X.shape  # M = number of ensemble members, N = state dimension
    n_obs = len(obs_idx)

    # Compute background mean
    mean_b = np.mean(X, axis=0)

    # Compute ensemble anomalies
    W = X - mean_b  # M x N

    # Compute background covariance P_b (N x N)
    P_b = (W.T @ W) / (M - 1)

    # Localization: apply a correlation-based taper to P_b
    # Use a Gaussian correlation function with range parameter
    # For Lorenz-96 with periodic boundary, distance is circular
    idx = np.arange(N)
    # Compute circular distance matrix
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, N - dist)  # circular distance

    # Correlation-based localization: taper = exp(-0.5 * (dist/range)^2)
    # Tuned range parameter for Lorenz-96
    range_param = 2.5
    taper = np.exp(-0.5 * (dist / range_param) ** 2)

    # Apply localization to background covariance
    P_loc = P_b * taper

    # Observation operator H: select observed indices
    # P_obs = P_loc[obs_idx, :][:, obs_idx] + R * I
    P_obs = P_loc[np.ix_(obs_idx, obs_idx)] + R * np.eye(n_obs)

    # Compute innovation: y - H mean_b
    innovation = y - mean_b[obs_idx]

    # Compute Kalman gain: K = P_loc H^T (H P_loc H^T + R)^-1
    # K is N x N_obs
    P_loc_Ht = P_loc[:, obs_idx]

    # Solve for K using Cholesky for numerical stability
    try:
        L = np.linalg.cholesky(P_obs)
        # K = P_loc_Ht @ inv(P_obs)
        Z = np.linalg.solve(L.T, P_loc_Ht.T)  # n_obs x N
        K = np.linalg.solve(L, Z).T  # N x n_obs
    except np.linalg.LinAlgError:
        try:
            K = P_loc_Ht @ np.linalg.inv(P_obs)
        except np.linalg.LinAlgError:
            K = P_loc_Ht @ np.linalg.pinv(P_obs)

    # Update mean: mean_a = mean_b + K @ innovation
    mean_a = mean_b + K @ innovation

    # Update ensemble: X_a = X + K @ (y - H X_b)
    H_X_b = X[:, obs_idx]
    innovation_ens = y[None, :] - H_X_b  # M x N_obs

    # update = innovation_ens @ K.T  # M x N
    update = innovation_ens @ K.T
    X_a = X + update

    # Ensure mean is preserved
    X_a = X_a - np.mean(X_a, axis=0) + mean_a

    # Adaptive inflation with more nuanced tuning
    # Calculate innovation variance per observed component
    innovation_var = np.var(innovation)
    
    # Expected innovation variance: mean of (ensemble variance at obs + R)
    ens_var_obs = np.var(X[:, obs_idx], axis=0)
    expected_innovation_var = np.mean(ens_var_obs + R)

    # Use a smoother inflation formula with a baseline
    if expected_innovation_var > 1e-10 and innovation_var > 1e-10:
        ratio = innovation_var / expected_innovation_var
        # Use sqrt for stability, with a floor to avoid under-inflation
        inflation = np.sqrt(max(ratio, 1.0))
        # Cap inflation to prevent divergence
        inflation = np.clip(inflation, 1.0, 1.15)
    else:
        inflation = 1.0

    # Apply scalar inflation to all components
    X_a = mean_a + inflation * (X_a - mean_a)

    # Add small stochastic perturbation to prevent filter collapse
    # This helps maintain ensemble spread in chaotic systems
    rng = np.random.default_rng(seed=42)
    noise_scale = 0.01 * np.std(X_a, axis=0)
    X_a = X_a + rng.normal(0.0, 1.0, size=X_a.shape) * noise_scale[None, :]

    # Ensure finite values and within bounds
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    X_a = np.clip(X_a, -100.0, 100.0)

    # Deadline check
    if time.time() > DEADLINE:
        return X_a

    return X_a
# EVOLVE-BLOCK-END
