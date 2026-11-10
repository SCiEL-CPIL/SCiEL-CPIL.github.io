import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    start_time = time.time()
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=int)
    
    # Compute ensemble mean and anomalies
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean
    
    # Covariance with regularization to handle small ensemble
    cov = (anomalies.T @ anomalies) / (M - 1)
    
    # Add small ridge regularization for numerical stability
    cov += 1e-6 * np.eye(N)
    
    # Gaspari-Cohn localization function
    def gaspari_cohn(d, L):
        """Gaspari-Cohn correlation function, periodic distance d, correlation length L."""
        d = np.abs(d)
        # Periodic distance
        d = np.minimum(d, N - d)
        if d <= L / 2:
            a = 5.0 / L
            b = 2.0 / L
            return (1.0 + a * d + b * d * d - (5.0/3.0) * a * b * d**3 + (1.0/3.0) * a * b * d**4) * \
                   (-2.0 * a * b * d**3 + (1.0/3.0) * a * b * d**4) / (1.0 + 0.0)
        elif d <= 1.5 * L:
            a = 5.0 / L
            b = 2.0 / L
            return (2.0 - b * d + (7.0/6.0) * a * b * d**2 - (7.0/30.0) * a * b * d**3 + \
                   (1.0/6.0) * a * b * d**4 - (1.0/40.0) * a * b * d**5)
        else:
            return 0.0
    
    # Use a tuned localization scale
    localization_scale = 5.0
    
    # Create distance matrix with periodic boundaries
    indices = np.arange(N)
    dist = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
    dist = np.minimum(dist, N - dist)
    
    # Apply Gaspari-Cohn localization
    loc_factor = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            loc_factor[i, j] = gaspari_cohn(dist[i, j], localization_scale)
    
    # Localized covariance
    cov_loc = cov * loc_factor
    
    # Observation operator
    H = np.zeros((K, N))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Slightly inflate observation variance to account for ensemble underdispersion
    R = observation_variance * 1.2 * np.eye(K)
    
    # Mean observation
    y_mean = np.mean(observations, axis=0)
    
    # Innovation
    innovation = y_mean - H @ mean
    
    # Innovation covariance
    S = H @ cov_loc @ H.T + R
    
    # Solve for gain
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Kalman gain
    gain = cov_loc @ H.T @ S_inv
    
    # Update mean
    mean_a = mean + gain @ innovation
    
    # Perturbed observation EnKF update for each member
    innovations = observations - ensemble[:, observed_indices]
    result = ensemble + innovations @ gain.T
    
    # Smooth adaptive inflation: continuous function of spread ratio
    ens_var_obs = np.var(ensemble[:, observed_indices], axis=0)
    obs_var = observation_variance
    
    # Smooth inflation factor using logistic-like function
    # Target spread ratio around 1.0, inflate if below, deflate if above
    spread_ratio = np.mean(ens_var_obs) / obs_var
    
    # Smooth inflation: alpha = 1 + beta * tanh((target - spread_ratio) / width)
    target_ratio = 0.8
    beta = 0.15
    width = 0.3
    inflation = 1.0 + beta * np.tanh((target_ratio - spread_ratio) / width)
    
    # Clamp inflation to reasonable range
    inflation = np.clip(inflation, 0.95, 1.15)
    
    # Apply inflation around updated mean
    result = mean_a + inflation * (result - mean_a)
    
    # Ensure finite and within bounds
    result = np.clip(result, -100, 100)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 110:
        return result
    
    return result
