"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Block-localized Ensemble Transform Kalman Filter (BLETKF) with
    adaptive spread control for Lorenz-96 partial-observation assimilation.
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
    mean_b = np.mean(X, axis=0)

    # Anomalies
    W = X - mean_b  # M x N

    # Block-localized transform EnKF
    # For N=20 periodic, use blocks of size B=8 with stride S=4
    # Blocks start at 0, 4, 8, 12, 16 (5 blocks), each covers 8 indices circularly
    B = 8  # block size
    S = 4  # stride
    block_starts = list(range(0, N, S))  # [0, 4, 8, 12, 16]

    # Accumulate contributions
    X_a = np.zeros_like(X)
    mean_a = np.zeros_like(mean_b)
    weight_sum = np.zeros(N)

    for start in block_starts:
        # Block indices (circular)
        block_idx = np.array([(start + j) % N for j in range(B)])

        # Which observed indices fall in this block?
        obs_in_block = [oi for oi in obs_idx if oi in block_idx]
        obs_in_block = np.array(obs_in_block, dtype=int)

        if len(obs_in_block) == 0:
            X_a[:, block_idx] += X[:, block_idx]
            mean_a[block_idx] += mean_b[block_idx]
            weight_sum[block_idx] += 1.0
            continue

        # Local anomalies for this block
        W_local = W[:, block_idx]  # M x B
        P_local = (W_local.T @ W_local) / (M - 1)  # B x B

        # Local observation covariance
        n_obs_local = len(obs_in_block)
        P_obs_local = P_local[np.ix_(obs_in_block, obs_in_block)] + R * np.eye(n_obs_local)

        # Innovation for this block
        innovation_local = y[obs_in_block] - mean_b[obs_in_block]

        # Compute local Kalman gain
        P_local_Ht = P_local[:, obs_in_block]  # B x n_obs_local

        try:
            L = np.linalg.cholesky(P_obs_local)
            Z = np.linalg.solve(L.T, P_local_Ht.T)
            K_local = np.linalg.solve(L, Z).T  # B x n_obs_local
        except np.linalg.LinAlgError:
            try:
                K_local = P_local_Ht @ np.linalg.inv(P_obs_local)
            except np.linalg.LinAlgError:
                K_local = P_local_Ht @ np.linalg.pinv(P_obs_local)

        # Update for this block
        H_X_local = X[:, block_idx][:, obs_in_block]  # M x n_obs_local
        innovation_ens_local = y[obs_in_block][None, :] - H_X_local
        update_local = innovation_ens_local @ K_local.T  # M x B
        X_a_block = X[:, block_idx] + update_local

        # Preserve local mean
        mean_a_block = mean_b[block_idx] + K_local @ innovation_local
        X_a_block = X_a_block - np.mean(X_a_block, axis=0) + mean_a_block

        # Accumulate
        X_a[:, block_idx] += X_a_block
        mean_a[block_idx] += mean_a_block
        weight_sum[block_idx] += 1.0

    # Average over overlapping blocks
    X_a /= weight_sum[None, :]
    mean_a /= weight_sum

    # Adaptive spread control
    innovation = y - mean_b[obs_idx]
    innovation_var = np.var(innovation)
    ens_var_obs = np.var(X[:, obs_idx], axis=0)
    expected_innovation_var = np.mean(ens_var_obs + R)

    if expected_innovation_var > 1e-10 and innovation_var > 1e-10:
        ratio = innovation_var / expected_innovation_var
        inflation = np.sqrt(ratio)
        inflation = np.clip(inflation, 1.0, 1.12)
    else:
        inflation = 1.0

    # Apply multiplicative inflation around the mean
    X_a = mean_a + inflation * (X_a - mean_a)

    # Ensure finite and bounded
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    X_a = np.clip(X_a, -100.0, 100.0)

    if time.time() > DEADLINE:
        return X_a

    return X_a
# EVOLVE-BLOCK-END
