"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for partial-observation Lorenz-96.
    
    Parameters:
    -----------
    ensemble : (16, 20) array - ensemble of 16 members, 20 state variables
    observations : (10,) or (16, 10) array - noisy observations of even-indexed coordinates
    observed_indices : array of 10 indices - the even indices [0, 2, 4, ..., 18]
    observation_variance : float - variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : (16, 20) array - updated ensemble
    """
    start_time = time.time()
    deadline = start_time + 115.0

    N, D = ensemble.shape
    M = len(observed_indices)

    obs_arr = np.asarray(observations, dtype=np.float64)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0].copy()
    else:
        obs_vec = obs_arr.flatten().copy()

    observed_indices = np.asarray(observed_indices, dtype=int)

    # --- Stochastic Ensemble Transform Kalman Filter ---
    # This approach uses a rank-reduced transformation that explicitly
    # enforces the ensemble spread constraint and avoids spurious correlations.

    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (N, D)

    # Observed anomalies
    obs_anomalies = anomalies[:, observed_indices]  # (N, M)

    # Compute innovation covariance S = H P H^T + R
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)
    R = observation_variance * np.eye(M)
    S = P_obs + R

    # Cross-covariance
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)  # (D, M)

    # Localization: minimum-image distance on ring
    d_idx = np.arange(D)
    loc_weights = np.ones((D, M))
    L = 2.0
    for m_idx, obs_i in enumerate(observed_indices):
        dist = np.abs(d_idx - obs_i)
        dist = np.minimum(dist, D - dist)
        loc_weights[:, m_idx] = np.exp(-0.5 * (dist / L) ** 2)

    PHt_loc = PHt * loc_weights

    # Compute S inverse
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    # Compute the gain matrix K = P H^T S^{-1}
    K = PHt_loc @ S_inv  # (D, M)

    # --- Stochastic Ensemble Transform ---
    # We want to find a transformation A such that:
    # 1. The updated ensemble has the correct mean
    # 2. The updated ensemble preserves the spread in observed directions
    #    (spread constraint)
    # 3. The transformation is close to identity (minimal change)

    # Innovation vector (mean innovation)
    innovation_mean = obs_vec - ens_mean[observed_indices]  # (M,)

    # First, compute the deterministic EKF update for the mean
    mean_update = K @ innovation_mean  # (D,)

    # For the stochastic transform, we work in anomaly space
    # The key idea: find A such that
    #   (anomalies @ A) has the same spread as the posterior
    # in the observation space, while being a minimal perturbation.

    # Posterior covariance in observation space:
    # P_post_obs = P_obs - P_obs @ S_inv @ P_obs
    # But we enforce this via the transform.

    # The innovation covariance for the anomalies:
    # S_anom = obs_anomalies @ obs_anomalies.T / (N-1) + R
    # = S (same as above)

    # We use the approach: A = I + C @ S^{-1/2} @ (something)
    # More specifically, we use the transform:
    # A = I - (obs_anomalies @ S_inv @ obs_anomalies.T) / (N-1) * beta
    # where beta is a relaxation parameter.

    # Actually, the proper stochastic ensemble transform uses:
    # Let Y = obs_anomalies (N x M)
    # We want the updated anomalies Y' = Y @ A_obs such that
    # Y' Y'^T / (N-1) = P_post_obs = P_obs - P_obs S_inv P_obs

    # Compute P_post_obs
    P_obs_mat = (obs_anomalies.T @ obs_anomalies) / (N - 1)  # (M, M)
    P_post_obs = P_obs_mat - P_obs_mat @ S_inv @ P_obs_mat  # (M, M)

    # Ensure P_post_obs is symmetric positive semi-definite
    P_post_obs = 0.5 * (P_post_obs + P_post_obs.T)

    # Factorize P_post_obs = L_post @ L_post.T (Cholesky with jitter)
    jitter = 1e-10
    P_post_obs_reg = P_post_obs + jitter * np.eye(M)
    try:
        L_post = np.linalg.cholesky(P_post_obs_reg)
    except np.linalg.LinAlgError:
        L_post = np.sqrt(np.maximum(np.diag(P_post_obs_reg), 0))
        L_post = np.diag(L_post)

    # Factorize P_obs = L_obs @ L_obs.T
    P_obs_reg = P_obs_mat + jitter * np.eye(M)
    try:
        L_obs = np.linalg.cholesky(P_obs_reg)
    except np.linalg.LinAlgError:
        L_obs = np.sqrt(np.maximum(np.diag(P_obs_reg), 0))
        L_obs = np.diag(L_obs)

    # The transform in observation space:
    # Y' = Y @ A_obs where A_obs satisfies Y @ A_obs @ (Y @ A_obs)^T / (N-1) = P_post_obs
    # If Y @ Y^T / (N-1) = P_obs = L_obs @ L_obs^T, then
    # A_obs = L_obs^{-1} @ L_post works in the limit N -> inf
    # For finite N, we use a regularized version.

    # Compute A_obs = pinv(L_obs) @ L_post
    try:
        A_obs = np.linalg.solve(L_obs, L_post)
    except np.linalg.LinAlgError:
        A_obs = np.linalg.pinv(L_obs) @ L_post

    # Now construct the full transform A (D x D) using the localization
    # The transform should be:
    # A = I - K @ (I - A_obs) @ H_eff
    # where H_eff is the localized observation operator

    # Actually, let's use a simpler and more robust approach:
    # Apply the transform directly in the full state space using the gain.

    # The ensemble transform is:
    # x'_i = x_i + K @ (y_i - y_obs)  for the deterministic part
    # Plus a stochastic perturbation to enforce spread.

    # Let's use the standard stochastic EKF formulation:
    # 1. Compute the gain K
    # 2. For each member, update: x_i' = x_i + K @ (y_i - y_obs)
    # 3. Add a random perturbation to enforce the spread constraint

    ens_obs = ensemble[:, observed_indices]  # (N, M)
    innovations = ens_obs - obs_vec[None, :]  # (N, M)

    # Deterministic update
    delta_x = innovations @ K.T  # (N, D)
    updated_ensemble = ensemble + delta_x

    # --- Spread enforcement via stochastic perturbation ---
    # The spread in observed variables should match P_post_obs
    # Current spread after deterministic update:
    updated_obs_anoms = updated_ensemble[:, observed_indices] - np.mean(updated_ensemble[:, observed_indices], axis=0)
    current_P_obs = (updated_obs_anoms.T @ updated_obs_anoms) / (N - 1)

    # We need to add perturbations to enforce P_post_obs
    # The residual covariance:
    residual = P_post_obs - current_P_obs
    residual = 0.5 * (residual + residual.T)

    # Add random perturbations in the observed space
    # The perturbation should have covariance residual
    if np.all(np.abs(residual) > 1e-12):
        # Cholesky of residual (with jitter)
        residual_reg = residual + jitter * np.eye(M)
        try:
            L_res = np.linalg.cholesky(residual_reg)
        except np.linalg.LinAlgError:
            L_res = np.sqrt(np.maximum(np.diag(residual_reg), 0))
            L_res = np.diag(L_res)

        # Generate random perturbations
        rand_pert = np.random.randn(N, M) @ L_res.T  # (N, M)
        # Scale to ensure correct covariance
        rand_pert = rand_pert / np.sqrt(N - 1) * np.sqrt(N - 1)  # already correct

        # Map back to full state using the gain
        perturbation_full = rand_pert @ K.T  # (N, D)
        updated_ensemble = updated_ensemble + perturbation_full

    # --- Adaptive inflation ---
    ens_spread_obs = np.std(updated_ensemble[:, observed_indices], axis=0, ddof=1)
    obs_noise_std = np.sqrt(observation_variance)
    mean_spread = np.mean(ens_spread_obs)

    ratio = mean_spread / obs_noise_std if obs_noise_std > 0 else 1.0
    target_ratio = 1.15
    inflation = 1.0 + 0.15 * (target_ratio - ratio) / max(ratio, 0.1)
    inflation = np.clip(inflation, 0.95, 1.10)

    updated_mean = np.mean(updated_ensemble, axis=0)
    updated_ensemble = updated_mean + inflation * (updated_ensemble - updated_mean)

    # --- Nonlinear correction via particle-weighted refinement ---
    ens_obs_updated = updated_ensemble[:, observed_indices]
    innovations_pf = obs_vec[None, :] - ens_obs_updated
    R_std = np.sqrt(observation_variance)
    log_weights = -0.5 * np.sum((innovations_pf / R_std) ** 2, axis=1)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= weights.sum()

    ess = 1.0 / np.sum(weights ** 2)

    if ess > N * 0.3:
        weighted_mean = np.sum(weights[:, None] * updated_ensemble, axis=0)
        blend = 0.20
        updated_mean2 = np.mean(updated_ensemble, axis=0)
        target_mean = (1 - blend) * updated_mean2 + blend * weighted_mean
        updated_ensemble = target_mean + (updated_ensemble - updated_mean2)
    else:
        r = np.random.uniform(0, 1.0 / N, 1)
        cum_weights = np.cumsum(weights)
        cum_weights[-1] = 1.0
        indices = np.searchsorted(cum_weights, r + np.arange(N) / N)
        indices = np.clip(indices, 0, N - 1)
        resampled = updated_ensemble[indices]
        ens_spread = np.std(updated_ensemble, axis=0)
        perturbation_scale = 0.05 * ens_spread
        resampled = resampled + np.random.randn(N, D) * perturbation_scale
        updated_ensemble = resampled

    # Deadline guard
    if time.time() > deadline:
        updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
        updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
        return updated_ensemble

    # Final safety checks
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)

    return updated_ensemble

# EVOLVE-BLOCK-END
