import numpy as np
import time

# Track global state across calls for adaptive inflation
_alpha_state = {'value': 1.02}
_call_count = 0

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with Localization and Adaptive Inflation.
    
    Args:
        ensemble: (16, 20) array of ensemble members.
        observations: (10,) array of observed values.
        observed_indices: (10,) array of indices observed.
        observation_variance: float, variance of observation noise.
        
    Returns:
        updated_ensemble: (16, 20) array of updated ensemble members.
    """
    global _alpha_state, _call_count
    _call_count += 1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    M, N = ensemble.shape
    obs_var = observation_variance
    H_indices = observed_indices
    n_obs = len(H_indices)
    
    # 1. Compute Mean and Anomalies
    mean_b = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean_b
    
    # 2. Localization: Gaspari-Cohn 5th order correlation
    def gaspari_cohn(d, sigma):
        d = np.abs(d)
        d = d / sigma
        c = np.zeros_like(d)
        mask1 = d <= 1
        c[mask1] = (-5/4 * d[mask1]**5 + 5/2 * d[mask1]**4 - 3/2 * d[mask1]**3 + 1)
        mask2 = (d > 1) & (d <= 2)
        c[mask2] = (1/12 * d[mask2]**5 - 5/4 * d[mask2]**4 + 5/2 * d[mask2]**3 - 5/2 * d[mask2]**2 + 4/3 * d[mask2] + 4/3)
        return c

    sigma = 4.0
    idx = np.arange(N)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, N - dist)
    C = gaspari_cohn(dist, sigma)
    
    # 3. Adaptive Inflation
    # Use previous alpha if available, otherwise start with 1.02
    alpha = _alpha_state['value']
    
    # Iterate to find optimal inflation
    for it in range(15):
        if time.time() > deadline:
            break
            
        # Inflate anomalies
        A_infl = anomalies * alpha
        P_b_infl = (A_infl.T @ A_infl) / (M - 1)
        P_b_loc_infl = P_b_infl * C
        
        # Observation space covariance
        P_obs_infl = P_b_loc_infl[np.ix_(H_indices, H_indices)]
        S_infl = P_obs_infl + obs_var * np.eye(n_obs)
        
        # Compute innovation
        nu = observations - mean_b[H_indices]
        
        # Solve for chi-squared
        try:
            L = np.linalg.cholesky(S_infl)
            tmp = np.linalg.solve(L, nu)
            chi2 = tmp @ tmp
        except np.linalg.LinAlgError:
            try:
                S_inv_infl = np.linalg.inv(S_infl)
                chi2 = nu @ S_inv_infl @ nu
            except np.linalg.LinAlgError:
                S_inv_infl = np.linalg.pinv(S_infl)
                chi2 = nu @ S_inv_infl @ nu
        
        # Update alpha
        alpha_new = np.sqrt(chi2 / n_obs)
        alpha_new = np.clip(alpha_new, 1.0, 2.0)
        
        # Damped update
        if it < 5:
            alpha = 0.5 * alpha + 0.5 * alpha_new
        else:
            alpha = 0.3 * alpha + 0.7 * alpha_new
        
        if abs(alpha - alpha_new) < 1e-4:
            break
    
    _alpha_state['value'] = alpha
    
    # 4. Final Update with optimal alpha
    A_infl = anomalies * alpha
    P_b_infl = (A_infl.T @ A_infl) / (M - 1)
    P_b_loc_infl = P_b_infl * C
    
    P_obs_infl = P_b_loc_infl[np.ix_(H_indices, H_indices)]
    S_infl = P_obs_infl + obs_var * np.eye(n_obs)
    
    # Compute gain K
    try:
        S_inv_infl = np.linalg.inv(S_infl)
    except np.linalg.LinAlgError:
        S_inv_infl = np.linalg.pinv(S_infl)
        
    P_b_loc_Ht_infl = P_b_loc_infl[:, H_indices]  # (20, 10)
    K = P_b_loc_Ht_infl @ S_inv_infl  # (20, 10)
    
    # Innovation
    nu = observations - mean_b[H_indices]
    
    # Update mean
    mean_a = mean_b + K @ nu
    
    # Update anomalies
    # x_m^a - mean_a = (I - K H) (x_m^{b, infl} - mean_b)
    H_A_infl = A_infl[:, H_indices]  # (16, 10)
    correction = H_A_infl @ K.T  # (16, 20)
    anomalies_a = A_infl - correction
    
    result = mean_a + anomalies_a
    
    # Ensure finite and within bounds
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
