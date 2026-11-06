"""
Ensemble Square Root Filter (EnSRF) with Adaptive Inflation and Localization
for Lorenz-96 assimilation.
"""
import numpy as np
import time

def _lorenz96(x, F):
    """Lorenz-96 dynamics: dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F"""
    n = len(x)
    dx = np.empty_like(x)
    for i in range(n):
        ip1 = (i + 1) % n
        im1 = (i - 1) % n
        im2 = (i - 2) % n
        dx[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + F
    return dx

def _rk4_step(x, dt, F):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96(x, F)
    k2 = _lorenz96(x + 0.5 * dt * k1, F)
    k3 = _lorenz96(x + 0.5 * dt * k2, F)
    k4 = _lorenz96(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Square Root Filter analysis with adaptive inflation and localization.
    
    Parameters:
        ensemble: (16, 20) array - ensemble of 16 members, 20 variables
        observations: (16, 10) array - noisy observations of even-indexed variables
        observed_indices: array of 10 indices - which variables are observed
        observation_variance: float - variance of observation noise
    
    Returns:
        (16, 20) array - updated ensemble
    """
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    M, N = ensemble.shape  # M=16 members, N=20 variables
    obs_idx = np.asarray(observed_indices, dtype=np.int64)
    n_obs = len(obs_idx)
    
    # Ensure inputs are numpy arrays
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    
    # 1. Compute ensemble mean and anomalies
    mean = ensemble.mean(axis=0)
    anomalies = ensemble - mean  # (M, N)
    
    # 2. Adaptive Inflation
    # We estimate the required inflation factor based on the innovation statistics.
    # Innovation: y - H(x)
    # H is the observation operator (extracts observed indices)
    obs_ens = ensemble[:, obs_idx]  # (M, n_obs)
    
    # Expected innovation covariance S_exp = H P H^T + R
    # P is the ensemble covariance
    # P_HH = (1/(M-1)) * obs_anomalies^T @ obs_anomalies
    obs_anoms = obs_ens - obs_ens.mean(axis=0)
    P_HH = (obs_anoms.T @ obs_anoms) / (M - 1)
    R = observation_variance * np.eye(n_obs)
    S_exp = P_HH + R
    
    # Observed innovation covariance S_obs
    # Innovations for each member: y - H(x_i)
    # Note: y is the same for all members in this context (deterministic obs)
    # But for inflation, we look at the spread of the ensemble around the mean innovation
    innov_mean = observations - obs_ens.mean(axis=0)
    innov_anoms = (obs_ens - obs_ens.mean(axis=0)) # This is - (y - Hx_i) - (y - Hx_mean) = Hx_mean - Hx_i
    # Actually, innovation anomalies are just the negative of the observation anomalies
    # S_obs_sample = (1/(M-1)) * innov_anoms^T @ innov_anoms
    S_obs_sample = (innov_anoms.T @ innov_anoms) / (M - 1)
    
    # Calculate ratio of traces to determine inflation
    trace_exp = np.trace(S_exp)
    trace_obs = np.trace(S_obs_sample)
    
    if trace_exp > 1e-12:
        ratio = trace_obs / trace_exp
        # If ratio < 1, ensemble is too spread out? No, if S_obs < S_exp, 
        # the innovations are smaller than expected, meaning the model spread might be too large 
        # OR the observation noise is larger. 
        # Standard adaptive inflation: if S_obs > S_exp, we need to inflate (spread more) to match.
        # If S_obs < S_exp, we might need to deflate (but we can't deflate below 1.0 usually to avoid collapse).
        # We use sqrt of ratio to adjust standard deviation.
        inflation_factor = np.sqrt(max(1.0, ratio))
        # Clip to prevent extreme inflation
        inflation_factor = np.clip(inflation_factor, 1.0, 1.5)
    else:
        inflation_factor = 1.0
    
    # Apply inflation to anomalies
    anomalies = anomalies * inflation_factor
    ensemble = mean + anomalies
    # Recompute mean (should be same) and anomalies
    mean = ensemble.mean(axis=0)
    anomalies = ensemble - mean
    
    # 3. Localization
    # Build localization matrix for N=20
    # Use Gaussian localization with length scale sigma_loc
    sigma_loc = 2.5
    loc_matrix = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            d = abs(i - j)
            d = min(d, N - d) # Periodic distance
            loc_matrix[i, j] = np.exp(-d**2 / (2.0 * sigma_loc**2))
    
    # 4. EnSRF Update
    # We perform a batch update using the localized covariance.
    # P = (1/(M-1)) * A^T A, where A is anomalies (M, N)
    P = (anomalies.T @ anomalies) / (M - 1)
    
    # Apply localization to the full covariance matrix
    P_loc = P * loc_matrix
    
    # Cross-covariance P_xH (N, n_obs)
    P_xH = P_loc[:, obs_idx]
    
    # Innovation covariance S = H P_loc H^T + R
    S = P_HH * 1.0 # Wait, P_HH was computed from original anomalies. 
    # We need to recompute S using the inflated and localized anomalies?
    # Yes, the update gain should be based on the inflated state.
    obs_anoms_inflated = obs_ens * inflation_factor # Approximation? 
    # Better: Recompute P_HH from the current anomalies (which are inflated)
    obs_anoms_curr = anomalies[:, obs_idx]
    P_HH_curr = (obs_anoms_curr.T @ obs_anoms_curr) / (M - 1)
    # Apply localization to P_HH? 
    # Localization is typically applied to the background covariance P.
    # So S = H (P * Loc) H^T + R
    # P_loc is already localized.
    # Extract the observed part of P_loc
    P_loc_HH = P_loc[np.ix_(obs_idx, obs_idx)]
    S = P_loc_HH + R
    
    # Calculate Gain K = P_xH S^{-1}
    # Solve S K^T = P_xH^T
    try:
        L = np.linalg.cholesky(S)
        # Solve L L^T X = P_xH^T
        X = np.linalg.solve(L.T, np.linalg.solve(L, P_xH.T))
        K = X.T # (N, n_obs)
    except np.linalg.LinAlgError:
        # Fallback
        S_reg = S + 1e-6 * np.eye(n_obs)
        K = P_xH @ np.linalg.inv(S_reg)
        
    # Update Mean
    innovation = observations - obs_ens.mean(axis=0) # Mean of H(x)
    mean_new = mean + K @ innovation
    
    # Update Ensemble (EnSRF style)
    # To avoid random perturbations, we use the square root form.
    # However, for a batch update, we can simply apply the linear gain to the anomalies.
    # x_i_new = x_i + K (y - H x_i)
    # = x_i + K (y - H mean) - K H (x_i - mean)
    # = mean_new - K H (x_i - mean)
    # Let's verify:
    # mean_new = mean + K (y - H mean)
    # x_i_new = mean_new + (x_i - mean) - K H (x_i - mean)
    # x_i_new = mean_new + (I - K H) (x_i - mean)
    
    # Compute K H
    # K is (N, n_obs), H is (n_obs, N)
    # We only need K H for the observed indices? No, K H is (N, N).
    # But we can compute the update for each member.
    # Anomalies are (M, N).
    # H (x_i - mean) is just the observed part of the anomaly: anomalies[:, obs_idx]
    # So, K H (x_i - mean) = K @ anomalies[:, obs_idx]
    
    # Update anomalies
    # New anomalies = (I - K H) @ Old Anomalies
    # Since we only have K @ (observed anomalies), we can do:
    # Delta = K @ anomalies[:, obs_idx]  # (M, N)
    # New Anomalies = Old Anomalies - Delta
    
    delta = anomalies[:, obs_idx] @ K.T # (M, n_obs) @ (n_obs, N) -> (M, N)
    new_anomalies = anomalies - delta
    
    # Construct new ensemble
    ensemble_new = mean_new[None, :] + new_anomalies
    
    # Ensure finiteness and bounds
    ensemble_new = np.clip(ensemble_new, -100.0, 100.0)
    ensemble_new = np.nan_to_num(ensemble_new, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    return ensemble_new
