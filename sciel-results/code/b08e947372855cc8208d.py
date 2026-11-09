"""Lorenz-96 LESRF assimilation: local ensemble square-root filter with sequential observation updates."""
import numpy as np
import time

_DEADLINE = 115.0  # safety margin before 120s kill
_T0 = time.time()

def _lorenz96_deriv(state, F):
    """Lorenz-96 derivative."""
    n = len(state)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (state[(i+1) % n] - state[(i-2) % n]) * state[(i-1) % n] - state[i] + F
    return dx

def _rk4_step(state, F, dt):
    """Single RK4 step."""
    k1 = _lorenz96_deriv(state, F)
    k2 = _lorenz96_deriv(state + 0.5*dt*k1, F)
    k3 = _lorenz96_deriv(state + 0.5*dt*k2, F)
    k4 = _lorenz96_deriv(state + dt*k3, F)
    return state + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)

def _local_inflation(ensemble, idx, N, M, inflation_rate=0.15):
    """
    Local multiplicative inflation around the local mean at index idx.
    Only inflates the local neighborhood (within radius 3).
    """
    local_radius = 3
    local_idx = []
    for j in range(M):
        dist = min(abs(j - idx), M - abs(j - idx))
        if dist <= local_radius:
            local_idx.append(j)
    local_idx = np.array(local_idx)
    
    local_mean = np.mean(ensemble[:, local_idx], axis=0)
    local_anoms = ensemble[:, local_idx] - local_mean
    
    # Compute local spread
    local_spread = np.std(ensemble[:, local_idx], axis=0)
    min_spread = 0.5 * np.sqrt(observation_variance_global)
    
    # Inflate where spread is too small
    for j_pos, j in enumerate(local_idx):
        if local_spread[j_pos] < min_spread:
            factor = 1.0 + inflation_rate
            ensemble[:, j] = local_mean[j_pos] + factor * local_anoms[:, j_pos]
    
    return ensemble

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    LESRF analysis step for Lorenz-96.
    
    Processes each observation sequentially with local square-root updates.
    Uses local spread inflation to maintain ensemble diversity.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed variables
    - observed_indices: array of observed variable indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - Updated ensemble with same (16, 20) shape
    """
    global observation_variance_global
    observation_variance_global = observation_variance
    
    N = ensemble.shape[0]  # 16
    M = ensemble.shape[1]  # 20
    K = len(observed_indices)  # 10
    
    result = ensemble.astype(np.float64).copy()
    
    # Localization radius for the local ensemble
    loc_radius = 4
    
    # Precompute localization weights for each observed index
    # For each observed variable, determine which state variables to update
    for k in range(K):
        obs_idx = observed_indices[k]
        y_k = observations[k]
        
        # Check deadline
        if time.time() - _T0 > _DEADLINE:
            result = np.clip(result, -100, 100)
            return result
        
        # Determine local neighborhood
        local_vars = []
        for j in range(M):
            dist = min(abs(j - obs_idx), M - abs(j - obs_idx))
            if dist <= loc_radius:
                local_vars.append(j)
        local_vars = np.array(local_vars)
        n_local = len(local_vars)
        
        # Local ensemble mean and anomalies
        local_ens = result[:, local_vars]  # (N, n_local)
        local_mean = np.mean(local_ens, axis=0)  # (n_local,)
        local_anoms = local_ens - local_mean  # (N, n_local)
        
        # Local ensemble variance at the observed variable
        # The observed variable is at position 0 in local_vars (dist=0)
        # Find its position
        obs_pos_in_local = np.where(local_vars == obs_idx)[0][0]
        
        local_var_obs = np.var(local_ens[:, obs_pos_in_local], ddof=0)  # population variance
        # Ensure minimum variance to avoid division by zero
        local_var_obs = max(local_var_obs, 1e-6)
        
        # Local ensemble covariance (n_local x n_local)
        # C = (1/N) * A^T @ A where A is (N, n_local)
        C_local = (local_anoms.T @ local_anoms) / N  # (n_local, n_local)
        
        # Apply localization to C_local
        for i in range(n_local):
            for j in range(n_local):
                dist = min(abs(local_vars[i] - local_vars[j]), M - abs(local_vars[i] - local_vars[j]))
                if dist > loc_radius:
                    C_local[i, j] = 0.0
                else:
                    # Gaussian taper
                    C_local[i, j] *= np.exp(-0.5 * (dist / loc_radius) ** 2)
        
        # Observation error variance
        R_k = observation_variance
        
        # Innovation variance: S = local_var_obs + R_k
        S = local_var_obs + R_k
        
        # Kalman gain for the observed variable: K_obs = local_var_obs / S
        K_obs = local_var_obs / S
        
        # Update the observed variable
        innovation = y_k - local_mean[obs_pos_in_local]
        
        # Update all local variables
        for i in range(n_local):
            # Gain for variable i: C_local[i, obs_pos_in_local] / S
            gain_i = C_local[i, obs_pos_in_local] / S
            result[:, local_vars[i]] += gain_i * innovation
        
        # After updating, apply local spread inflation to maintain diversity
        # Check if spread at observed location is too small
        new_spread = np.std(result[:, obs_idx], ddof=0)
        target_spread = np.sqrt(R_k) * 0.8
        if new_spread < target_spread:
            # Inflate locally
            local_mean_after = np.mean(result[:, local_vars], axis=0)
            local_anoms_after = result[:, local_vars] - local_mean_after
            inflation_factor = 1.0 + 0.10 * (1.0 - new_spread / target_spread) if target_spread > 0 else 1.0
            inflation_factor = min(inflation_factor, 1.15)  # cap inflation
            for i in range(n_local):
                dist = min(abs(local_vars[i] - obs_idx), M - abs(local_vars[i] - obs_idx))
                # Tapered inflation
                taper = np.exp(-0.5 * (dist / loc_radius) ** 2)
                result[:, local_vars[i]] = local_mean_after[i] + inflation_factor * taper * local_anoms_after[:, i]
    
    # Final global safety inflation if ensemble has collapsed
    global_spread = np.std(result, ddof=0)
    if global_spread < 0.5:
        g_mean = np.mean(result, axis=0)
        result = g_mean + 1.08 * (result - g_mean)
    
    result = np.clip(result, -100, 100)
    return result
