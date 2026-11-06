"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Localized Ensemble Kalman Filter with adaptive inflation and
    a high-precision polish pass for Lorenz-96 state reconstruction.
    Uses a single localized covariance update followed by a second
    pass with a tighter localization range, blended for robustness.
    """
    t_start = time.time()
    DEADLINE = 145.0

    X = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R = float(observation_variance)

    M, N = X.shape
    n_obs = len(obs_idx)

    mean_b = np.mean(X, axis=0)
    W = X - mean_b
    P_b = (W.T @ W) / (M - 1)

    idx = np.arange(N)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, N - dist)

    # Pass 1: wider localization
    range1 = 2.0
    taper1 = np.exp(-0.5 * (dist / range1) ** 2)
    P_loc1 = P_b * taper1
    P_obs1 = P_loc1[np.ix_(obs_idx, obs_idx)] + R * np.eye(n_obs)
    innovation = y - mean_b[obs_idx]
    P_loc_Ht1 = P_loc1[:, obs_idx]

    try:
        L1 = np.linalg.cholesky(P_obs1)
        Z1 = np.linalg.solve(L1.T, P_loc_Ht1.T)
        K1 = np.linalg.solve(L1, Z1).T
    except np.linalg.LinAlgError:
        try:
            K1 = P_loc_Ht1 @ np.linalg.inv(P_obs1)
        except np.linalg.LinAlgError:
            K1 = P_loc_Ht1 @ np.linalg.pinv(P_obs1)

    mean_a1 = mean_b + K1 @ innovation
    innovation_ens = y[None, :] - X[:, obs_idx]
    X_a1 = X + innovation_ens @ K1.T
    X_a1 = X_a1 - np.mean(X_a1, axis=0) + mean_a1

    # Adaptive inflation
    innovation_var = np.var(innovation)
    ens_var_obs = np.var(X[:, obs_idx], axis=0)
    expected_innovation_var = np.mean(ens_var_obs + R)
    if expected_innovation_var > 0 and innovation_var > 0:
        inflation = np.sqrt(innovation_var / expected_innovation_var)
        inflation = np.clip(inflation, 1.0, 1.25)
    else:
        inflation = 1.0
    X_a1 = mean_a1 + inflation * (X_a1 - mean_a1)

    X_a1 = np.nan_to_num(X_a1, nan=0.0, posinf=100.0, neginf=-100.0)
    X_a1 = np.clip(X_a1, -100.0, 100.0)

    if time.time() > DEADLINE:
        return X_a1

    # Pass 2: tighter localization for polish
    if time.time() < DEADLINE - 5.0:
        range2 = 1.8
        taper2 = np.exp(-0.5 * (dist / range2) ** 2)
        P_loc2 = P_b * taper2
        P_obs2 = P_loc2[np.ix_(obs_idx, obs_idx)] + R * np.eye(n_obs)
        P_loc_Ht2 = P_loc2[:, obs_idx]

        try:
            L2 = np.linalg.cholesky(P_obs2)
            Z2 = np.linalg.solve(L2.T, P_loc_Ht2.T)
            K2 = np.linalg.solve(L2, Z2).T
        except np.linalg.LinAlgError:
            try:
                K2 = P_loc_Ht2 @ np.linalg.inv(P_obs2)
            except np.linalg.LinAlgError:
                K2 = K1

        mean_a2 = mean_b + K2 @ innovation
        X_a2 = X + innovation_ens @ K2.T
        X_a2 = X_a2 - np.mean(X_a2, axis=0) + mean_a2

        alpha = 0.5
        X_a = alpha * X_a1 + (1.0 - alpha) * X_a2
        X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
        X_a = np.clip(X_a, -100.0, 100.0)
    else:
        X_a = X_a1

    return X_a
# EVOLVE-BLOCK-END
