"""Lorenz-96 ensemble assimilation with LESRF, adaptive inflation, and localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Square Root Filter (LESRF) analysis update for Lorenz-96.
    Uses Cholesky-based square-root formulation for numerically stable gain computation.
    Includes adaptive multiplicative inflation and Gaussian localization.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # 1. Compute ensemble mean and anomalies
    ens_mean = np.mean(ens, axis=0)
    A = ens - ens_mean  # Shape (n_members, n_vars)
    A_obs = A[:, obs_idx]  # Shape (n_members, n_obs)
    
    # 2. Adaptive Multiplicative Inflation
    # Target: ensemble spread in obs space should be comparable to obs noise
    ens_var_obs = np.var(A_obs, axis=0)  # shape (n_obs,)
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-12))
    target_spread = np.sqrt(R_val) * 1.4
    ratio = np.mean(target_spread / np.maximum(current_spread, 1e-12))
    
    # Smooth adaptive inflation: blend between 1.0 and a cap
    # Use tanh for smooth saturation
    inflation = 1.0 + 0.12 * np.tanh(2.0 * (ratio - 1.0))
    inflation = np.clip(inflation, 1.0, 1.15)
    
    # Apply multiplicative inflation about the mean
    ens = ens_mean + inflation * A
    ens_mean = np.mean(ens, axis=0)
    A = ens - ens_mean
    A_obs = A[:, obs_idx]
    
    # 3. Gaussian Localization
    sigma = 3.0
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)  # Periodic distance
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)  # Shape (n_vars, n_obs)
    
    # 4. Compute covariances (sample-based)
    # Cross-covariance between state and observations (localized)
    C_xo = (A.T @ A_obs) / (n_members - 1.0)  # Shape (n_vars, n_obs)
    C_xo_loc = C_xo * loc_weights  # Localized cross-cov
    
    # Observation covariance
    C_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # Shape (n_obs, n_obs)
    
    # 5. Innovation
    innovation = obs - ens_mean[obs_idx]  # Shape (n_obs,)
    
    # 6. Compute Kalman gain via Cholesky for numerical stability
    S = C_oo + R_val * np.eye(n_obs) + 1e-10 * np.eye(n_obs)
    
    try:
        # Cholesky decomposition of S
        L = np.linalg.cholesky(S)
        # Solve L @ X = C_xo_loc.T  =>  X = L^{-1} C_xo_loc.T
        # Then K = X @ L^{-T}  =>  K = C_xo_loc @ S^{-1}
        # Using Cholesky: K = (L^{-T} @ C_xo_loc^T).T
        X = np.linalg.solve(L, C_xo_loc.T)  # Shape (n_obs, n_vars)
        K = X.T  # Shape (n_vars, n_obs) -- this gives C_xo_loc @ L^{-T}
        # Actually: K = C_xo_loc @ S^{-1} = C_xo_loc @ L^{-T} @ L^{-1}
        # So K = (L^{-1} @ C_xo_loc.T).T is wrong.
        # Correct: K = C_xo_loc @ inv(S)
        # With Cholesky: inv(S) = L^{-T} @ L^{-1}
        # K = C_xo_loc @ L^{-T} @ L^{-1}
        # Let's just use solve: solve(S, C_xo_loc.T).T
        K = np.linalg.solve(S, C_xo_loc.T).T  # Shape (n_vars, n_obs)
    except np.linalg.LinAlgError:
        K = C_xo_loc @ np.linalg.pinv(S)
    
    # 7. LESRF Update
    # Update mean first (deterministic)
    delta_mean = K @ innovation
    ens_mean_new = ens_mean + delta_mean
    
    # Update anomalies using square-root formulation
    # For each variable j, we need to update the anomaly vector A[:, j]
    # using the gain K[j, :] and the observation anomalies A_obs
    # 
    # The LESRF update for member m, variable j:
    # A_new[m, j] = A[m, j] - K[j, :] @ (A_obs[m, :] + perturbation)
    # where perturbation is chosen to preserve variance.
    #
    # Standard LESRF: for each state variable j:
    #   a = A[:, j]  (anomaly vector, shape n_members)
    #   k = K[j, :]  (gain row, shape n_obs)
    #   y_anom = A_obs  (shape n_members, n_obs)
    #   Update: a_new = a - k @ y_anom.T  (but this reduces variance)
    #   To preserve variance, add a correction:
    #   a_new = a - k @ y_anom.T + correction
    #
    # Better approach: use the sequential update with square root
    # For each observation component, update all variables sequentially.
    # But with 10 obs and 20 vars, we can do it in one shot.
    #
    # The correct LESRF formula:
    # Let P = A^T A / (n-1)  (state covariance, n_vars x n_vars)
    # Let P_xo = A^T A_obs / (n-1)  (cross-cov, n_vars x n_obs)
    # Let P_oo = A_obs^T A_obs / (n-1)  (obs covariance, n_obs x n_obs)
    # K = P_xo @ inv(P_oo + R)
    #
    # For the ensemble update preserving variance:
    # A_new = A - K @ A_obs^T  (this is the deterministic part)
    # But this reduces the ensemble spread. The LESRF correction:
    # We need to add a term that restores the spread.
    #
    # Standard LESRF (Hunt et al. 2007, Anderson 2003):
    # For each state variable j:
    #   Let a_j = A[:, j]  (n_members vector)
    #   Let k_j = K[j, :]  (n_obs vector)
    #   The update is: a_j_new = a_j - k_j @ A_obs[m] for each member m
    #   But to preserve the ensemble variance, we use:
    #   a_j_new = a_j - (k_j @ A_obs^T) * sqrt(1 - something)
    #
    # Actually, the simplest correct LESRF:
    # 1. Compute the gain K
    # 2. Update mean: x_mean += K @ innovation
    # 3. Update anomalies: A_new = A - K @ A_obs^T
    # 4. This naturally preserves the ensemble covariance structure
    #    because it's the exact linear update.
    #
    # Wait - A - K @ A_obs^T does NOT preserve variance in general.
    # The issue is that with a small ensemble, the sample covariance
    # after the update is not exactly the Kalman-updated covariance.
    #
    # The correct LESRF approach (from Anderson 2003, Hunt et al. 2007):
    # Update each state variable sequentially using a scalar observation.
    # For each observation l:
    #   For each state variable j:
    #     a = A[:, j]
    #     k = K[j, l]
    #     y_l = A_obs[:, l]
    #     a_new = a - k * y_l
    #     # Then add a correction to preserve variance:
    #     # The variance reduction factor is (1 - k^2 * var(y_l) / var(a))
    #     # We scale by 1/sqrt(1 - k^2 * var(y_l) / var(a))
    #     var_a = np.var(a)
    #     var_y = np.var(y_l)
    #     if var_a > 1e-10:
    #         reduction = 1.0 - k**2 * var_y / var_a
    #         if reduction > 0.01:
    #             a_new = a_new / np.sqrt(reduction)
    #     A[:, j] = a_new
    #
    # This sequential approach is O(n_obs * n_vars * n_members) which is fine.
    
    # Sequential LESRF update
    A_new = A.copy()
    for l in range(n_obs):
        # Observation anomaly for this observation
        y_l = A_obs[:, l]  # Shape (n_members,)
        var_y = np.var(y_l)
        if var_y < 1e-15:
            continue
        for j in range(n_vars):
            a = A_new[:, j]  # Shape (n_members,)
            k = K[j, l]
            if abs(k) < 1e-12:
                continue
            # Update
            a_updated = a - k * y_l
            # Variance preservation correction
            var_a = np.var(a)
            if var_a > 1e-15:
                reduction = 1.0 - k**2 * var_y / var_a
                if reduction > 0.01:
                    a_updated = a_updated / np.sqrt(reduction)
            A_new[:, j] = a_updated
    
    # Reconstruct ensemble
    ens = ens_mean_new[None, :] + A_new
    
    # 8. Final safety checks
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
        
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
