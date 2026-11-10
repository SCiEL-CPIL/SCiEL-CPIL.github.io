import numpy as np

def _gaspari_cohn(d, c=5.0):
    """
    Gaspari-Cohn correlation function for localization.
    d: array of distances
    c: correlation length parameter
    """
    d = np.abs(d)
    result = np.zeros_like(d)
    
    # d <= 0.5 * c
    mask1 = d <= 0.5 * c
    if np.any(mask1):
        dm = d[mask1]
        cm = c
        result[mask1] = (-5.0/4.0) * (dm**5 / cm**5) + 5.0 * (dm**4 / cm**4) - 10.0 * (dm**3 / cm**3) + 5.0 * dm
        
    # 0.5 * c < d <= 1.5 * c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    if np.any(mask2):
        dm = d[mask2]
        cm = c
        result[mask2] = (1.0/12.0) * (dm**5 / cm**5) - (5.0/4.0) * (dm**4 / cm**4) + (5.0/2.0) * (dm**3 / cm**3) - 5.0 * (dm**2 / cm**2) + (25.0/12.0) * (dm / cm) + 1.0/2.0
        
    # 1.5 * c < d <= 2.5 * c
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    if np.any(mask3):
        dm = d[mask3]
        cm = c
        result[mask3] = (-1.0/16.0) * (dm**5 / cm**5) + (1.0/4.0) * (dm**4 / cm**4) - (15.0/8.0) * (dm**3 / cm**3) + (5.0/2.0) * (dm**2 / cm**2) - (25.0/12.0) * (dm / cm) + 25.0/12.0

    # d > 2.5 * c
    # result remains 0
    
    return result

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    EnKF with localization and adaptive inflation.
    """
    M, N = ensemble.shape
    n_obs = len(observed_indices)
    
    # 1. Compute ensemble mean and anomalies
    x_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - x_mean
    
    # 2. Extract observed anomalies
    obs_anomalies = anomalies[:, observed_indices]
    
    # 3. Compute covariance matrices
    # P_oo: (n_obs, n_obs)
    P_oo = (obs_anomalies @ obs_anomalies.T) / (M - 1)
    
    # P_xo: (N, n_obs)
    P_xo = (anomalies @ obs_anomalies.T) / (M - 1)
    
    # 4. Compute Kalman Gain K = P_xo (P_oo + R)^-1
    # R is diagonal with observation_variance
    R = np.eye(n_obs) * observation_variance
    
    # Solve (P_oo + R) K^T = P_xo^T to avoid explicit inverse
    # Or just use np.linalg.solve
    try:
        # K^T = (P_oo + R)^-1 P_xo^T
        K_T = np.linalg.solve(P_oo + R, P_xo.T)
        K = K_T.T
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if singular
        K_T = np.linalg.pinv(P_oo + R) @ P_xo.T
        K = K_T.T

    # 5. Apply Localization to K
    # K has shape (N, n_obs)
    # We need to multiply K[i, j] by L(distance(i, observed_indices[j]))
    # Create distance matrix
    # i: 0..N-1, j: 0..n_obs-1
    # dist = min(|i - obs_idx|, N - |i - obs_idx|)
    i_idx = np.arange(N)[:, np.newaxis]
    j_idx = observed_indices[np.newaxis, :]
    
    dists = np.abs(i_idx - j_idx)
    dists = np.minimum(dists, N - dists)
    
    L = _gaspari_cohn(dists, c=5.0)
    
    K_loc = K * L
    
    # 6. Update Ensemble
    # Innovation: y - H x_f
    # H x_f is ensemble[:, observed_indices]
    H_x = ensemble[:, observed_indices]
    innovations = observations[:, np.newaxis] - H_x # (n_obs, M)
    
    # Update: X_a = X + K_loc @ innovations.T ?
    # x_a,k = x_f,k + K_loc @ innovations[:,k]
    # X_a = X + K_loc @ innovations.T
    updated_ensemble = ensemble + K_loc @ innovations.T
    
    # 7. Adaptive Inflation
    # Compute variance of innovations
    innov_var = np.var(innovations, axis=1) # (n_obs,)
    
    # Expected innovation variance: Var(H x_f) + R
    # Var(H x_f) is diagonal of H P_f H^T
    # P_f is full covariance, but we only have P_oo for observed part
    # Var(H x_f) = diag(P_oo)
    pred_innov_var = np.diag(P_oo) + observation_variance
    
    # Avoid division by zero
    pred_innov_var = np.maximum(pred_innov_var, 1e-10)
    
    # Inflation factor per observed variable? Or global?
    # Global is simpler and often sufficient
    # alpha^2 = mean(innov_var) / mean(pred_innov_var)
    # But we want to correct the ensemble spread.
    # If innov_var > pred_innov_var, ensemble is under-dispersed.
    ratio = np.mean(innov_var) / np.mean(pred_innov_var)
    
    # Clip inflation to avoid wild swings
    # If ratio > 1, inflate. If ratio < 1, we might deflate or do nothing.
    # Standard adaptive inflation usually only inflates or uses a smoother formula.
    # Let's use: alpha = sqrt(max(ratio, 1.0)) ?
    # Or a softer version: alpha = sqrt(ratio)
    # To be safe, let's only inflate if ratio is significantly > 1.
    # And limit the inflation factor.
    if ratio > 1.0:
        alpha = np.sqrt(ratio)
        # Limit alpha to prevent blow up
        alpha = np.clip(alpha, 1.0, 1.5)
    else:
        alpha = 1.0
    
    # Apply inflation around the mean
    x_mean_a = np.mean(updated_ensemble, axis=0)
    updated_ensemble = x_mean_a + alpha * (updated_ensemble - x_mean_a)
    
    # 8. Safety clip
    updated_ensemble = np.clip(updated_ensemble, -100, 100)
    
    # Ensure finite
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
