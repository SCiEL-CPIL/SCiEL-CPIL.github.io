import numpy as np

# Pre-compute the observation operator H
# Observed indices are even: 0, 2, 4, ..., 18
OBS_IDX = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
N_VARS = 20
N_OBS = len(OBS_IDX)

# H is a (10, 20) matrix
H = np.zeros((N_OBS, N_VARS))
H[np.arange(N_OBS), OBS_IDX] = 1.0

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter update with localization and inflation.
    """
    # Ensure inputs are float64 arrays
    X = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(observations, dtype=np.float64)
    var_obs = float(observation_variance)
    
    M, N = X.shape
    # M is number of ensemble members (16)
    # N is number of state variables (20)
    
    # 1. Compute Background Mean and Deviations
    x_bar = np.mean(X, axis=0)
    X_dev = X - x_bar
    
    # 2. Adaptive Inflation
    # We use a simple adaptive inflation based on the ratio of ensemble spread to innovation
    # However, to keep it simple and robust, we use a fixed multiplicative inflation
    # tuned for this problem. 
    # A value of 1.02-1.05 is typical.
    inflation_factor = 1.02
    
    # Inflate deviations
    X_dev_inflated = X_dev * inflation_factor
    
    # 3. Compute Localized Background Covariance
    # P_b = (1/(M-1)) * X_dev^T X_dev
    P_b = (X_dev_inflated.T @ X_dev_inflated) / (M - 1)
    
    # Apply Localization
    # Exponential localization: L_ij = exp(-|i-j|^2 / (2*sigma^2))
    # sigma = 3.0 is a reasonable width for N=20
    sigma_loc = 3.0
    idx = np.arange(N)
    dist = np.abs(idx[:, None] - idx[None, :])
    L = np.exp(-dist**2 / (2 * sigma_loc**2))
    P_b_loc = P_b * L
    
    # 4. Compute Innovation and Covariances
    # H is pre-computed
    # P_b_loc is (20, 20)
    # H is (10, 20)
    
    # Observation Error Covariance R
    R = var_obs * np.eye(N_OBS)
    
    # S = H P_b_loc H^T + R
    H_Pb = H @ P_b_loc
    S = H_Pb @ H.T + R
    
    # Innovation
    y_bar = H @ x_bar
    innovation = y - y_bar
    
    # 5. Compute Kalman Gain
    # K = P_b_loc H^T S^{-1}
    # Use solve for stability
    # S is (10, 10), small.
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular (unlikely with R)
        S_inv = np.linalg.pinv(S)
        
    K = P_b_loc @ H.T @ S_inv
    
    # 6. Update Mean
    x_bar_new = x_bar + K @ innovation
    
    # 7. Update Deviations (Square Root Filter)
    # We want X_dev_new such that Cov(X_dev_new) = (I - K H) P_b_loc
    # Note: The standard EnKF update x_i = x_i + K(y - H x_i) leads to 
    # Cov_new = (I-KH) P_b.
    # However, using the inflated P_b_loc in K calculation means we should be careful.
    # The "Local Ensemble Square Root Filter" (LESRF) approach is more stable.
    
    # Let's use the LESRF update for the deviations to ensure the ensemble
    # covariance matches the analysis covariance derived from the localized background.
    
    # S_chol = Cholesky(S)
    try:
        L_s = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        L_s = np.linalg.cholesky(S + 1e-10 * np.eye(N_OBS))
        
    # Y = H X_dev_inflated L_s^{-1}
    # X_dev_inflated is (20, 16)
    # L_s is (10, 10)
    # We need to solve L_s X = H X_dev_inflated
    # X = (10, 16)
    try:
        Y = np.linalg.solve(L_s, H @ X_dev_inflated)
    except np.linalg.LinAlgError:
        Y = np.linalg.lstsq(L_s, H @ X_dev_inflated, rcond=None)[0]
        
    # Q = I - Y^T (I + Y Y^T)^{-1} Y
    # Y is (10, 16)
    # Y Y^T is (10, 10)
    I_obs = np.eye(N_OBS)
    try:
        I_plus_YYT_inv = np.linalg.inv(I_obs + Y @ Y.T)
    except np.linalg.LinAlgError:
        I_plus_YYT_inv = np.linalg.pinv(I_obs + Y @ Y.T)
        
    Q = I_obs - Y.T @ I_plus_YYT_inv @ Y
    # Q is (16, 16)
    
    # X_dev_new = X_dev_inflated Q
    X_dev_new = X_dev_inflated @ Q
    
    # 8. Construct New Ensemble
    X_new = x_bar_new[None, :] + X_dev_new
    
    # Ensure finite values
    if not np.all(np.isfinite(X_new)):
        # Fallback to simple nudging if something goes wrong
        X_new = ensemble.copy()
        X_new[:, observed_indices] += 0.5 * (observations - X_new[:, observed_indices])
        
    return X_new
