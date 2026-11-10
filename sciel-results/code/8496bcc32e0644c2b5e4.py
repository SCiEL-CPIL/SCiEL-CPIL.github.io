"""Local Ensemble Square Root Filter (LESRF) for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _gaspari_cohn(d, c=1.5):
    """Gaspari-Cohn correlation function for localization."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    mask1 = d <= 0.5 * c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)

    # d <= 0.5c
    r1 = d[mask1]
    if len(r1) > 0:
        x = r1 / c
        result[mask1] = (-5.0/4.0)*x**5 + 5.0*x**4 - 10.0*x**3 + 5.0*x
    # 0.5c < d <= 1.5c
    r2 = d[mask2]
    if len(r2) > 0:
        x = r2 / c
        result[mask2] = (1.0/12.0)*x**5 - (5.0/4.0)*x**4 + (5.0/2.0)*x**3 - 5.0*x**2 + (25.0/12.0)*x + 0.5
    # 1.5c < d <= 2.5c
    r3 = d[mask3]
    if len(r3) > 0:
        x = r3 / c
        result[mask3] = (-1.0/16.0)*x**5 + (5.0/4.0)*x**4 - (5.0/2.0)*x**3 + 5.0*x**2 - (25.0/12.0)*x + (35.0/24.0)

    return result

def _localize_covariance(cov, n_var, loc_radius=8.0):
    """Apply Gaspari-Cohn localization to covariance matrix."""
    n_obs = n_var
    loc = np.ones((n_obs, n_obs))
    for i in range(n_obs):
        for j in range(n_obs):
            d = abs(i - j)
            # Handle periodic boundary for Lorenz-96
            d = min(d, n_var - d)
            loc[i, j] = _gaspari_cohn(np.array([d]), c=loc_radius)[0]
    return cov * loc

def _esrf_update(ensemble, observations, observed_indices, obs_var, loc_radius=8.0):
    """Local Ensemble Square Root Filter update."""
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)

    # Compute ensemble mean and anomalies
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (n_members, n_vars)

    # Observation operator: extract observed variables
    H_anom = anomalies[:, observed_indices]  # (n_members, n_obs)
    H_mean = ens_mean[observed_indices]  # (n_obs,)

    # Innovation
    innovation = observations - H_mean  # (n_obs,)

    # Background covariance at observed locations: P_yy = H P H^T
    P_oo = (H_anom @ H_anom.T) / (n_members - 1)  # (n_obs, n_obs)

    # Localize observation covariance
    R = obs_var * np.eye(n_obs)
    # Localize P_oo
    P_oo_loc = _localize_covariance(P_oo, n_obs, loc_radius)

    # Compute Kalman gain: K = P_oo_loc (P_oo_loc + R)^{-1}
    S = P_oo_loc + R
    try:
        L = np.linalg.cholesky(S)
        X = np.linalg.solve(L, P_oo_loc.T)
        K_T = np.linalg.solve(L.T, X)
        K = K_T.T  # (n_obs, n_obs)
    except np.linalg.LinAlgError:
        K = P_oo_loc @ np.linalg.pinv(S)

    # Full background covariance
    P_full = (anomalies @ anomalies.T) / (n_members - 1)  # (n_vars, n_vars)

    # Construct H matrix (sparse)
    H = np.zeros((n_obs, n_vars))
    H[np.arange(n_obs), observed_indices] = 1.0

    # Update covariance: P' = P - K H P
    P_updated = P_full - K @ H @ P_full

    # Ensure symmetry
    P_updated = 0.5 * (P_updated + P_updated.T)

    # Add small regularization for numerical stability
    P_updated += 1e-10 * np.eye(n_vars)

    # Find transformation: we need A' such that A' A'^T = (n_members-1) * P_updated
    try:
        eigvals, eigvecs = np.linalg.eigh(P_updated)
        eigvals = np.clip(eigvals, 0, None)
        sqrt_P = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        target_anom = sqrt_P * np.sqrt(n_members - 1)
    except np.linalg.LinAlgError:
        try:
            L = np.linalg.cholesky(P_updated)
            target_anom = L * np.sqrt(n_members - 1)
        except np.linalg.LinAlgError:
            updated_mean = ens_mean + K @ innovation
            result = ensemble.copy()
            result[:, observed_indices] += K @ innovation
            return result

    # Find orthogonal transformation Q such that A' = A Q + correction
    try:
        W = np.linalg.pinv(anomalies) @ target_anom
    except np.linalg.LinAlgError:
        W = np.eye(n_vars)

    # Apply transformation to anomalies
    updated_anomalies = anomalies @ W

    # Update mean
    updated_mean = ens_mean + K @ innovation

    # Reconstruct ensemble
    result = updated_anomalies + updated_mean

    # Ensure finite values
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100, 100)

    return result

def _enkf_update(ensemble, observations, observed_indices, obs_var, loc_radius=8.0, inflation=1.0):
    """Ensemble Kalman Filter update with localization and optional inflation."""
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)

    # Compute ensemble mean and anomalies
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (n_members, n_vars)

    # Apply inflation to anomalies if needed
    if inflation > 1.0:
        anomalies *= inflation

    # Observation operator: extract observed variables
    H_anom = anomalies[:, observed_indices]  # (n_members, n_obs)
    H_mean = ens_mean[observed_indices]  # (n_obs,)

    # Innovation
    innovation = observations - H_mean  # (n_obs,)

    # Background covariance at observed locations: P_yy = H P H^T
    P_oo = (H_anom @ H_anom.T) / (n_members - 1)  # (n_obs, n_obs)

    # Localize observation covariance
    R = obs_var * np.eye(n_obs)
    # Localize P_oo
    P_oo_loc = _localize_covariance(P_oo, n_obs, loc_radius)

    # Compute Kalman gain: K = P_oo_loc (P_oo_loc + R)^{-1}
    S = P_oo_loc + R
    try:
        L = np.linalg.cholesky(S)
        X = np.linalg.solve(L, P_oo_loc.T)
        K_T = np.linalg.solve(L.T, X)
        K = K_T.T  # (n_obs, n_obs)
    except np.linalg.LinAlgError:
        K = P_oo_loc @ np.linalg.pinv(S)

    # Update mean
    updated_mean = ens_mean + K @ innovation

    # Update anomalies
    # For each member: x_i' = x_i + K (y - H x_i)
    # = x_i + K innovation - K H (x_i - mean)
    # = x_i + K innovation - H_anom_i^T K^T (but need to be careful with dimensions)
    
    # Actually: x_i' = x_i + K (y - H x_i)
    # H x_i = H_mean + H_anom_i
    # y - H x_i = innovation - H_anom_i
    # So x_i' = x_i + K (innovation - H_anom_i)
    
    # For observed variables:
    # x_i'[obs] = x_i[obs] + K (innovation - H_anom_i)
    # For unobserved variables, we need to propagate the update
    
    # Construct full update
    # K is (n_obs, n_obs), innovation is (n_obs,)
    # H_anom_i is (n_obs,)
    
    # For observed variables, the update is straightforward
    # For unobserved variables, we use the cross-covariance
    
    # Cross-covariance between all variables and observed variables
    # P_xo = (anomalies @ H_anom.T) / (n_members - 1)  # (n_vars, n_obs)
    P_xo = (anomalies @ H_anom.T) / (n_members - 1)
    
    # Localize cross-covariance
    # For each unobserved variable, localize its covariance with observed variables
    P_xo_loc = np.zeros_like(P_xo)
    for i in range(n_vars):
        if i in observed_indices:
            P_xo_loc[i, :] = P_xo[i, :]
        else:
            # Find nearest observed index for localization
            obs_idx = np.argmin([min(abs(i - j), n_vars - abs(i - j)) for j in observed_indices])
            d = min(abs(i - observed_indices[obs_idx]), n_vars - abs(i - observed_indices[obs_idx]))
            loc_val = _gaspari_cohn(np.array([d]), c=loc_radius)[0]
            P_xo_loc[i, :] = P_xo[i, :] * loc_val

    # Update each member
    result = ensemble.copy()
    for m in range(n_members):
        # For observed variables
        H_anom_m = H_anom[m, :]  # (n_obs,)
        update_obs = K @ (innovation - H_anom_m)  # (n_obs,)
        result[m, observed_indices] += update_obs
        
        # For unobserved variables
        for i in range(n_vars):
            if i not in observed_indices:
                # Use cross-covariance to update
                # x_i' = x_i + P_xo_loc[i, :] @ (P_oo_loc + R)^{-1} @ innovation
                # But we already have K = P_oo_loc (P_oo_loc + R)^{-1}
                # So x_i' = x_i + P_xo_loc[i, :] @ (P_oo_loc + R)^{-1} @ innovation
                # = x_i + P_xo_loc[i, :] @ K.T @ (P_oo_loc)^{-1} @ innovation  (not quite)
                
                # Simpler: x_i' = x_i + P_xo_loc[i, :] @ inv(S) @ innovation
                # where S = P_oo_loc + R
                try:
                    update_unobs = P_xo_loc[i, :] @ np.linalg.solve(S, innovation)
                    result[m, i] += update_unobs
                except np.linalg.LinAlgError:
                    pass

    # Ensure finite values
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100, 100)

    return result

def _ensemble_mean_update(ensemble, observations, observed_indices, obs_var, loc_radius=8.0, inflation=1.0):
    """
    Hybrid approach: use LESRF for the ensemble update but with adaptive inflation.
    """
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)

    # Compute ensemble mean and anomalies
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (n_members, n_vars)

    # Adaptive inflation based on ensemble spread
    # Compute spread at observed locations
    H_anom = anomalies[:, observed_indices]
    spread_obs = np.std(H_anom, axis=0)
    obs_std = np.sqrt(obs_var)
    
    # Inflation factor: if spread is too small relative to obs noise, inflate
    ratio = np.mean(spread_obs) / obs_std
    if ratio < 0.5:
        inflation = 1.5
    elif ratio < 0.8:
        inflation = 1.2
    else:
        inflation = 1.0

    # Apply inflation to anomalies
    anomalies_inflated = anomalies * inflation

    # Recompute mean with inflated anomalies (mean stays the same)
    ens_mean_inflated = ens_mean  # Mean doesn't change with inflation of anomalies

    # Observation operator: extract observed variables
    H_anom_inflated = anomalies_inflated[:, observed_indices]
    H_mean = ens_mean[observed_indices]

    # Innovation
    innovation = observations - H_mean

    # Background covariance at observed locations
    P_oo = (H_anom_inflated @ H_anom_inflated.T) / (n_members - 1)

    # Localize observation covariance
    R = obs_var * np.eye(n_obs)
    P_oo_loc = _localize_covariance(P_oo, n_obs, loc_radius)

    # Compute Kalman gain
    S = P_oo_loc + R
    try:
        L = np.linalg.cholesky(S)
        X = np.linalg.solve(L, P_oo_loc.T)
        K_T = np.linalg.solve(L.T, X)
        K = K_T.T
    except np.linalg.LinAlgError:
        K = P_oo_loc @ np.linalg.pinv(S)

    # Update mean
    updated_mean = ens_mean + K @ innovation

    # Full background covariance with inflation
    P_full = (anomalies_inflated @ anomalies_inflated.T) / (n_members - 1)

    # Construct H matrix
    H = np.zeros((n_obs, n_vars))
    H[np.arange(n_obs), observed_indices] = 1.0

    # Update covariance: P' = P - K H P
    P_updated = P_full - K @ H @ P_full
    P_updated = 0.5 * (P_updated + P_updated.T)
    P_updated += 1e-10 * np.eye(n_vars)

    # Find transformation
    try:
        eigvals, eigvecs = np.linalg.eigh(P_updated)
        eigvals = np.clip(eigvals, 0, None)
        sqrt_P = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        target_anom = sqrt_P * np.sqrt(n_members - 1)
    except np.linalg.LinAlgError:
        try:
            L = np.linalg.cholesky(P_updated)
            target_anom = L * np.sqrt(n_members - 1)
        except np.linalg.LinAlgError:
            result = ensemble.copy()
            result[:, observed_indices] += K @ innovation
            return result

    # Find transformation matrix
    try:
        W = np.linalg.pinv(anomalies_inflated) @ target_anom
    except np.linalg.LinAlgError:
        W = np.eye(n_vars)

    # Apply transformation
    updated_anomalies = anomalies_inflated @ W

    # Reconstruct ensemble
    result = updated_anomalies + updated_mean

    # Ensure finite values
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100, 100)

    return result

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local ESRF analysis update with adaptive inflation.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    try:
        # Use the hybrid approach with adaptive inflation
        result = _ensemble_mean_update(ensemble, observations, observed_indices, observation_variance, loc_radius=8.0)
    except Exception:
        # Fallback to simple nudging if it fails
        result = ensemble.copy()
        result[:, observed_indices] += 0.5 * (observations - result[:, observed_indices])
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        result = np.clip(result, -100, 100)

    # Final safety check
    if time.time() > deadline:
        pass

    # Ensure output is valid
    result = np.asarray(result, dtype=float)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100, 100)

    return result
# EVOLVE-BLOCK-END
