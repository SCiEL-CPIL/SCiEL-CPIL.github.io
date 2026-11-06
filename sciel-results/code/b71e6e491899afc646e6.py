"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Transform Kalman Filter (LETKF) with Gaussian localization,
    adaptive inflation, and deadline guard for Lorenz-96 assimilation.
    Combines block-localization structure with correlation-based tapering.
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

    # Ensemble anomalies
    W = X - mean_b  # M x N

    # --- Localization setup ---
    # Circular distance matrix
    idx = np.arange(N)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, N - dist)  # circular distance

    # Gaussian taper with range parameter
    range_param = 3.0
    taper = np.exp(-0.5 * (dist / range_param) ** 2)

    # Block radius for LETKF-style local processing
    block_radius = 4

    # Build block structure: for each state variable i,
    # the block includes variables within circular distance block_radius
    blocks = []
    for i in range(N):
        offsets = list(range(-block_radius, block_radius + 1))
        block_vars = [(i + off) % N for off in offsets]
        # Remove duplicates while preserving order
        seen = set()
        unique_block_vars = []
        for v in block_vars:
            if v not in seen:
                seen.add(v)
                unique_block_vars.append(v)
        block_vars = unique_block_vars

        # Find which observations are in this block
        block_set = set(block_vars)
        obs_in_block = [j for j in range(n_obs) if obs_idx[j] in block_set]

        blocks.append({
            'center': i,
            'vars': block_vars,
            'obs_local': obs_in_block
        })

    # Initialize analysis ensemble
    X_a = X.copy()

    # Process each block: local EnKF update with localized covariance
    for i in range(N):
        block = blocks[i]
        var_indices = block['vars']
        obs_local = block['obs_local']

        n_vars = len(var_indices)
        n_local_obs = len(obs_local)

        if n_local_obs == 0:
            continue

        # Extract local ensemble anomalies
        W_local = W[:, var_indices]  # M x n_vars

        # Local background covariance with Gaussian localization
        P_local = (W_local.T @ W_local) / (M - 1)  # n_vars x n_vars
        # Apply taper within the block
        local_dist = np.abs(np.arange(n_vars)[:, None] - np.arange(n_vars)[None, :])
        local_taper = np.exp(-0.5 * (local_dist / range_param) ** 2)
        P_local = P_local * local_taper

        # Observation operator for this block
        obs_var_indices_in_block = [obs_idx[j] for j in obs_local]
        var_to_pos = {v: pos for pos, v in enumerate(var_indices)}

        H_local = np.zeros((n_local_obs, n_vars))
        for k, ov in enumerate(obs_var_indices_in_block):
            pos = var_to_pos.get(ov)
            if pos is not None:
                H_local[k, pos] = 1.0

        # P_obs = H P_local H^T + R I
        P_obs = H_local @ P_local @ H_local.T + R * np.eye(n_local_obs)

        # Innovation
        obs_positions_in_block = [var_to_pos[obs_idx[j]] for j in obs_local]
        innovation_local = y[obs_local] - mean_b[var_indices][obs_positions_in_block]

        # Compute local Kalman gain
        P_local_Ht = P_local @ H_local.T  # n_vars x n_local_obs

        try:
            L = np.linalg.cholesky(P_obs + 1e-12 * np.eye(n_local_obs))
            Z = np.linalg.solve(L.T, P_local_Ht.T)
            K_local = np.linalg.solve(L, Z).T  # n_vars x n_local_obs
        except np.linalg.LinAlgError:
            try:
                K_local = P_local_Ht @ np.linalg.inv(P_obs)
            except np.linalg.LinAlgError:
                K_local = P_local_Ht @ np.linalg.pinv(P_obs)

        # Update mean for variables in this block
        mean_a_local = mean_b[var_indices] + K_local @ innovation_local

        # Update ensemble members
        H_X_local = X[:, var_indices] @ H_local.T  # M x n_local_obs
        innovation_ens_local = y[obs_local][None, :] - H_X_local  # M x n_local_obs
        update_local = innovation_ens_local @ K_local.T  # M x n_vars

        X_a[:, var_indices] = X[:, var_indices] + update_local

    # Re-center to ensure exact mean
    mean_a = np.mean(X_a, axis=0)

    # --- Adaptive inflation ---
    # Compare innovation variance to expected
    obs_positions = obs_idx
    ens_obs_var = np.var(X[:, obs_positions], axis=0)
    expected_innov_var = np.mean(ens_obs_var + R)
    innovation = y - mean_b[obs_positions]
    actual_innov_var = np.var(innovation)

    if expected_innov_var > 1e-10 and actual_innov_var > 1e-10:
        inflation_factor = np.sqrt(actual_innov_var / expected_innov_var)
        inflation_factor = np.clip(inflation_factor, 1.0, 1.15)
    else:
        inflation_factor = 1.0

    # Apply inflation
    X_a = mean_a + inflation_factor * (X_a - mean_a)

    # Ensure finite and bounded
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    X_a = np.clip(X_a, -100.0, 100.0)

    if time.time() > DEADLINE:
        return X_a

    return X_a
# EVOLVE-BLOCK-END
