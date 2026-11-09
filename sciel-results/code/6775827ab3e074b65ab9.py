import numpy as np
import time

# Module-level state for adaptive inflation
_inflation_factor = 1.0
_localization_length = 3.0

def _lorenz96_rhs(x, F):
    """Compute RHS of Lorenz-96 system."""
    xm2 = np.roll(x, 2)
    xm1 = np.roll(x, 1)
    xp1 = np.roll(x, -1)
    return (xp1 - xm2) * xm1 - x + F

def _rk4_step(x, dt, F):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _get_localization_matrix(n, length):
    """
    Generate Gaussian localization matrix.
    Distance is circular distance on the ring of n points.
    """
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(i - j)
            d = min(d, n - d)
            dist[i, j] = d
    
    L = np.exp(-0.5 * (dist / length) ** 2)
    return L

def analyze(ensemble, observations, observed_indices, observation_variance):
    global _inflation_factor
    
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n_members, n_vars = ensemble.shape
    result = ensemble.copy()
    
    # 1. Adaptive Inflation
    ens_mean_obs = np.mean(result[:, observed_indices], axis=0)
    ens_var_obs = np.var(result[:, observed_indices], axis=0, ddof=1)
    
    innovation = observations - ens_mean_obs
    innov_var = np.mean(innovation ** 2)
    
    # Target ratio: (ens_var + obs_var) / innov_var
    # If ratio < 1, ensemble is too tight -> inflate
    ratios = (innov_var) / (np.mean(ens_var_obs) + observation_variance)
    
    alpha = 0.1
    if ratios > 1.5:
        _inflation_factor *= (1.0 + alpha)
    elif ratios < 0.5:
        _inflation_factor *= (1.0 - alpha)
    
    # Clamp inflation factor to reasonable bounds
    _inflation_factor = np.clip(_inflation_factor, 0.5, 5.0)
    
    # Apply inflation to the ensemble (centered on mean)
    mean_all = np.mean(result, axis=0)
    result = mean_all + _inflation_factor * (result - mean_all)
    
    # Check time
    if time.time() > deadline - 5.0:
        return result
    
    # 2. Localization Matrix
    L = _get_localization_matrix(n_vars, _localization_length)
    
    # 3. LES-RF Update
    m = np.mean(result, axis=0)
    anomalies = result - m
    
    # Covariance C = (1/(N-1)) * anomalies.T @ anomalies
    C = (anomalies.T @ anomalies) / (n_members - 1)
    
    # Apply Localization to C
    C_loc = C * L
    
    # Compute H C_loc H^T
    C_oo = C_loc[np.ix_(observed_indices, observed_indices)]
    
    # R
    R = observation_variance * np.eye(len(observed_indices))
    
    # Innovation
    y = observations
    m_obs = m[observed_indices]
    d = y - m_obs
    
    # Compute inverse of (C_oo + R)
    S = C_oo + R
    try:
        # Cholesky
        L_s = np.linalg.cholesky(S)
        X = np.linalg.solve(L_s, d)
        Y = np.linalg.solve(L_s.T, X)
        S_inv_d = Y
    except np.linalg.LinAlgError:
        S_inv_d = np.linalg.pinv(S) @ d
    
    # Kalman Gain K = C_loc H^T S^-1
    C_h = C_loc[:, observed_indices]
    
    # Correction for mean
    correction = C_h @ S_inv_d
    m_a = m + correction
    
    # Update Ensemble Anomalies using Linear EKF Update (Simpler and Robust)
    # x_i^a = x_i + K (y - H x_i)
    # K = C_loc H^T S^-1
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        
    K = C_h @ S_inv  # 20x10
    
    # Compute d for all members
    x_obs = result[:, observed_indices].T  # 10x16
    d_all = observations[:, np.newaxis] - x_obs  # 10x16
    
    # Correction
    corr_all = K @ d_all  # 20x16
    
    # Update ensemble
    result = result + corr_all
    
    # Check time
    if time.time() > deadline - 2.0:
        return result
    
    # 4. Final Inflation Check
    # Re-calculate ensemble variance and adjust inflation if needed for next time?
    # We already did this at the start.
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result
