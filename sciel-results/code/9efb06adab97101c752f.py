import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with localization and fixed inflation for Lorenz-96.
    """
    start_time = time.time()
    
    # Convert to numpy arrays for efficiency
    X_b = np.array(ensemble, dtype=np.float64)
    y_obs = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)
    
    M, N = X_b.shape  # M: ensemble size, N: state dimension
    
    # 1. Compute background mean and perturbations
    mean_b = np.mean(X_b, axis=0)
    W_b = X_b - mean_b
    
    # 2. Compute background covariance (N x N)
    P_b = (W_b.T @ W_b) / (M - 1)
    
    # 3. Localization
    idx = np.arange(N)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, N - dist)
    
    # Exponential localization kernel
    sigma_loc = 2.0
    L_matrix = np.exp(-dist**2 / (2 * sigma_loc**2))
    
    P_b_loc = P_b * L_matrix
    
    # 4. Observation operator H (sparse)
    P_oo = P_b_loc[np.ix_(obs_idx, obs_idx)]
    R = R_val * np.eye(len(obs_idx))
    
    S = P_oo + R
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        
    K_full = P_b_loc[:, obs_idx] @ S_inv
    
    # 5. Update the mean
    H_mean_b = mean_b[obs_idx]
    innovation = y_obs - H_mean_b
    
    mean_a = mean_b + K_full @ innovation
    
    # 6. Update the perturbations
    HW_b = W_b[obs_idx, :]
    KH_Wb = K_full @ HW_b
    
    W_a = W_b - KH_Wb
    
    # 7. Inflation
    # Fixed multiplicative inflation
    alpha = 1.02
    W_a = W_a * alpha
    
    # 8. Reconstruct ensemble
    X_a = mean_a[None, :] + W_a
    
    # 9. Safety checks
    if not np.all(np.isfinite(X_a)):
        return np.array(ensemble, dtype=np.float64)
    
    X_a = np.clip(X_a, -100, 100)
    
    if time.time() - start_time > 10:
         return np.array(ensemble, dtype=np.float64)

    return X_a
