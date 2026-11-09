"""Lorenz-96 ensemble assimilation with optimized EnSRF."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Optimized ensemble assimilation update for Lorenz-96.
    Uses ensemble square-root filter with adaptive inflation and localization.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.asarray(observed_indices)
    R = float(observation_variance) if np.isscalar(observation_variance) else float(np.mean(observation_variance))

    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)

    # --- Adaptive multiplicative inflation ---
    # Start with mild inflation, adjust based on innovation spread
    inflation = 1.03

    # Apply initial inflation
    ens_mean = ens.mean(axis=0)
    ens = ens_mean + inflation * (ens - ens_mean)
    ens_mean = ens.mean(axis=0)

    # --- Localization weights (Gaussian on periodic ring) ---
    # L96 correlations decay over ~4-5 grid points
    corr_length = 4.5
    # Precompute distance matrix
    idx = np.arange(n_vars)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    # Weight for each observation to each state variable
    # shape: (n_obs, n_vars)
    loc_w = np.exp(-0.5 * (dist[np.ix_(obs_idx, idx)] / corr_length) ** 2)

    # --- Compute ensemble anomalies ---
    anomalies = ens - ens_mean  # (n_members, n_vars)
    # Ensemble covariance proxy: (1/(M-1)) * A^T A
    # We'll use the rank-one form for efficiency

    alpha = n_members / (n_members - 1.0)

    # --- Process all observations jointly using a localized EnSRF ---
    # For each observation j, update all variables i with weight w[j,i]
    # Vectorized over observations where possible

    H_ens = ens[:, obs_idx]  # (n_members, n_obs)
    H_mean = H_ens.mean(axis=0)  # (n_obs,)
    innovations = obs - H_mean  # (n_obs,)

    # Compute observation-space ensemble variance for each obs
    H_anom = H_ens - H_mean  # (n_members, n_obs)
    # S_j = alpha * sum_m (H_anom[m,j])^2 / (M-1) + R
    S = alpha * np.sum(H_anom ** 2, axis=0) / (n_members - 1.0) + R  # (n_obs,)

    # Avoid division by zero
    S = np.maximum(S, 1e-12)

    # --- Joint update: for each state variable i, compute weighted sum over observations ---
    # cross-cov(i, j) = alpha * sum_m anomalies[m,i] * H_anom[m,j] / (M-1)
    # shape: (n_vars, n_obs)
    cross_cov = alpha * (anomalies.T @ H_anom) / (n_members - 1.0)  # (n_vars, n_obs)

    # Apply localization: cross_cov_loc[i,j] = cross_cov[i,j] * loc_w[j,i]
    # loc_w shape (n_obs, n_vars), so we need loc_w.T (n_vars, n_obs)
    cross_cov_loc = cross_cov * loc_w.T  # (n_vars, n_obs)

    # Update: ens[:, i] += sum_j (cross_cov_loc[i,j] / S[j]) * (obs[j] - H_ens[:,j])
    # = sum_j cross_cov_loc[i,j] * innovations[j] / S[j]  ... but H_ens[:,j] varies per member
    # Actually: ens[:, i] += sum_j cross_cov_loc[i,j] / S[j] * innovations[j]
    # Wait, the standard EnSRF sequential update is:
    #   x_m += K_i * (y_j - H x_m)  where K_i = cross_cov(i,j)/S
    # But for a joint (non-sequential) update we can use:
    #   x_m += sum_j [cross_cov(i,j)/S_j] * (y_j - H_j(x_m))
    # = sum_j [cross_cov(i,j)/S_j] * y_j - sum_j [cross_cov(i,j)/S_j] * H_j(x_m)

    # Let's do it properly: for each member m and variable i:
    #   delta[m,i] = sum_j cross_cov_loc[i,j] / S[j] * (obs[j] - H_ens[m,j])
    # = sum_j cross_cov_loc[i,j] / S[j] * obs[j] - sum_j cross_cov_loc[i,j] / S[j] * H_ens[m,j]

    # Precompute: gain[i,j] = cross_cov_loc[i,j] / S[j]
    gain = cross_cov_loc / S[None, :]  # (n_vars, n_obs)

    # delta[m,i] = gain[i,:] @ (obs - H_ens[m,:])
    # = gain[i,:] @ obs - gain[i,:] @ H_ens[m,:]

    # Compute: delta = (obs @ gain.T) - (H_ens @ gain.T)  ... shapes?
    # obs @ gain.T: (n_obs,) @ (n_obs, n_vars) -> (n_vars,)
    # H_ens @ gain.T: (n_members, n_obs) @ (n_obs, n_vars) -> (n_members, n_vars)

    # Actually: delta[m,i] = sum_j gain[i,j] * (obs[j] - H_ens[m,j])
    # = sum_j gain[i,j]*obs[j] - sum_j gain[i,j]*H_ens[m,j]

    # term1[i] = sum_j gain[i,j] * obs[j]  -> gain @ obs  (n_vars,)
    term1 = gain @ obs  # (n_vars,)

    # term2[m,i] = sum_j gain[i,j] * H_ens[m,j] -> H_ens @ gain.T  (n_members, n_vars)
    term2 = H_ens @ gain.T  # (n_members, n_vars)

    delta = term1[None, :] - term2  # (n_members, n_vars)
    ens = ens + delta

    # --- Check time, apply final inflation if needed ---
    if time.time() > deadline - 2.0:
        # Quick cleanup and return
        ens = np.clip(ens, -100.0, 100.0)
        ens = np.where(np.isfinite(ens), ens, 0.0)
        return ens.astype(ensemble.dtype)

    # --- Adaptive inflation based on innovation statistics ---
    # After update, check if ensemble spread is too small relative to innovation
    ens_mean_new = ens.mean(axis=0)
    # Compute post-update innovation spread for observed variables
    H_ens_new = ens[:, obs_idx]
    H_mean_new = H_ens_new.mean(axis=0)
    post_innovations = obs - H_mean_new
    # Estimate observation-space variance from ensemble
    post_H_anom = H_ens_new - H_mean_new
    post_S_ens = alpha * np.sum(post_H_anom ** 2, axis=0) / (n_members - 1.0)

    # If ensemble variance is much smaller than expected, inflate more
    # Expected: post_S_ens should be ~ R * (1 - K) where K is gain
    # Simple heuristic: if mean post_S_ens < 0.3 * R, inflate
    mean_post_var = np.mean(post_S_ens)
    if mean_post_var < 0.5 * R:
        extra_inflation = 1.05
    elif mean_post_var < 1.0 * R:
        extra_inflation = 1.02
    else:
        extra_inflation = 1.0

    ens = ens_mean_new + extra_inflation * (ens - ens_mean_new)

    # --- Final cleanup ---
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)

    if time.time() > deadline:
        return ens.astype(ensemble.dtype)

    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
