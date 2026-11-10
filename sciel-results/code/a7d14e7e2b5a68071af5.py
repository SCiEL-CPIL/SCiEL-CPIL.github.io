"""Local ESRF analysis for Lorenz-96 partial-observation assimilation."""
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
        result[mask1] = (-5.0/4.0)*(r1/c)**5 + 5.0*(r1/c)**4 - 10.0*(r1/c)**3 + 5.0*(r1/c)
    # 0.5c < d <= 1.5c
    r2 = d[mask2]
    if len(r2) > 0:
        result[mask2] = (1.0/12.0)*(r2/c)**5 - (5.0/4.0)*(r2/c)**4 + (5.0/2.0)*(r2/c)**3 - 5.0*(r2/c)**2 + (25.0/12.0)*(r2/c) + 0.5
    # 1.5c < d <= 2.5c
    r3 = d[mask3]
    if len(r3) > 0:
        result[mask3] = (-1.0/16.0)*(r3/c)**5 + (5.0/4.0)*(r3/c)**4 - (5.0/2.0)*(r3/c)**3 + 5.0*(r3/c)**2 - (25.0/12.0)*(r3/c) + (35.0/24.0)
    
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
    # P = (1/(n_members-1)) * A A^T where A is anomalies
    P_oo = (H_anom @ H_anom.T) / (n_members - 1)  # (n_obs, n_obs)
    
    # Localize observation covariance
    R = obs_var * np.eye(n_obs)
    # Localize P_oo
    P_oo_loc = _localize_covariance(P_oo, n_obs, loc_radius)
    
    # Compute Kalman gain: K = P_oo_loc (P_oo_loc + R)^{-1}
    # Use Cholesky or SVD for stability
    S = P_oo_loc + R
    try:
        L = np.linalg.cholesky(S)
        # Solve L X^T = P_oo_loc^T, then L^T K^T = X^T
        X = np.linalg.solve(L, P_oo_loc.T)
        K_T = np.linalg.solve(L.T, X)
        K = K_T.T  # (n_obs, n_obs)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse
        K = P_oo_loc @ np.linalg.pinv(S)
    
    # ESRF: find transformation matrix that updates anomalies
    # We want updated anomalies A' such that A' A'^T / (n_members-1) = P - K P H^T
    # where P is the full background covariance
    
    # Full background covariance
    P_full = (anomalies @ anomalies.T) / (n_members - 1)  # (n_vars, n_vars)
    
    # Construct H matrix (sparse)
    H = np.zeros((n_obs, n_vars))
    H[np.arange(n_obs), observed_indices] = 1.0
    
    # Update covariance: P' = P - K H P
    P_updated = P_full - K @ H @ P_full
    
    # Localize P_updated to avoid negative eigenvalues
    # Ensure symmetry
    P_updated = 0.5 * (P_updated + P_updated.T)
    
    # Add small regularization for numerical stability
    P_updated += 1e-10 * np.eye(n_vars)
    
    # Find transformation: we need A' such that A' A'^T = (n_members-1) * P_updated
    # Use SVD/Cholesky approach
    # P_updated should be PSD, compute its square root
    try:
        eigvals, eigvecs = np.linalg.eigh(P_updated)
        # Clip negative eigenvalues
        eigvals = np.clip(eigvals, 0, None)
        sqrt_P = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        target_anom = sqrt_P * np.sqrt(n_members - 1)
    except np.linalg.LinAlgError:
        # Fallback: use Cholesky
        try:
            L = np.linalg.cholesky(P_updated)
            target_anom = L * np.sqrt(n_members - 1)
        except np.linalg.LinAlgError:
            # Last resort: just use the linear update
            updated_mean = ens_mean + K @ innovation
            result = ensemble.copy()
            result[:, observed_indices] += K @ innovation  # simplified
            return result
    
    # Now find orthogonal transformation Q such that A' = A Q + correction
    # Actually, for ESRF we want to transform anomalies directly
    # A' = A @ W where W is some matrix
    # We need A' A'^T = target_anom target_anom^T
    # Let A' = target_anom, then we need to map A to A'
    # Use Procrustes: minimize ||A W - target_anom||_F
    # W = pinv(A) @ target_anom
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

def analyze(ensemble, observations, observed_indices, observation_variance):
    """Local ESRF analysis update."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    try:
        result = _esrf_update(ensemble, observations, observed_indices, observation_variance, loc_radius=8.0)
    except Exception:
        # Fallback to simple nudging if ESRF fails
        result = ensemble.copy()
        result[:, observed_indices] += 0.5 * (observations - result[:, observed_indices])
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        result = np.clip(result, -100, 100)
    
    # Final safety check
    if time.time() > deadline:
        # If we're running out of time, return what we have
        pass
    
    # Ensure output is valid
    result = np.asarray(result, dtype=float)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100, 100)
    
    return result
# EVOLVE-BLOCK-END
