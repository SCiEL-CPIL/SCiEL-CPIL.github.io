"""Lorenz-96 ensemble assimilation: Ensemble Square Root Filter (EnSRF)
with sequential coordinate-wise regression, adaptive inflation, and
localization via spatially-weighted regression coefficients."""
import numpy as np
import time

# Module-level adaptive state
_inflation = 1.05
_inflation_alpha = 0.75
_inflation_min = 1.0
_inflation_max = 2.5
_loc_range = 3.0

# Deadline guard
_DEADLINE = 115.0  # seconds, with 5s safety margin before 120s limit
_start_time = None


def _lorenz96_deriv(x, F):
    if x.ndim == 1:
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    else:
        return (np.roll(x, -1, axis=1) - np.roll(x, 2, axis=1)) * np.roll(x, 1, axis=1) - x + F


def _localization_weight(dist, dim, rng):
    """Gaussian localization weight for periodic domain."""
    d = min(abs(dist), dim - abs(dist))
    return np.exp(-0.5 * (d / rng) ** 2)


def analyze(ensemble, observations, observed_indices, observation_variance):
    global _inflation, _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    t0 = time.time()
    N, dim = ensemble.shape
    n_obs = len(observed_indices)
    
    # Sanitize inputs
    ensemble = np.where(np.isfinite(ensemble), ensemble, 0.0).copy()
    observations = np.where(np.isfinite(observations), observations, 0.0).astype(float)
    
    # --- Step 1: Inflation ---
    mean = ensemble.mean(axis=0)
    anomalies = ensemble - mean
    inflated = mean + _inflation * anomalies
    
    # --- Step 2: Sequential EnSRF (coordinate-wise regression) ---
    # Process each observed coordinate sequentially.
    # For each observed coordinate j, regress all unobserved coordinates onto it
    # using the ensemble covariance, with localization.
    
    # Precompute: for each observed index, which other indices are "unobserved"
    # Actually, in EnSRF we update all coordinates (observed and unobserved) sequentially.
    # The standard approach: for each observation, update the ensemble by regressing
    # each state variable onto the observed variable.
    
    # We'll use the "ensemble adjustment" form:
    # For each observed variable y_j:
    #   For each state variable x_i:
    #     Compute regression coefficient a_ij = Cov(x_i, y_j) / Var(y_j)
    #     Apply localization: a_ij *= exp(-0.5 * (dist(i,j)/loc_range)^2)
    #     Update: x_i += a_ij * (y_j - y_j_forecast)
    #     Then update y_j_forecast for subsequent use (but in sequential EnSRF,
    #     we just update the ensemble and move on)
    
    # Actually, the proper sequential EnSRF (like the one in Evensen 1998, or
    # the "ensemble square root filter" by Anderson 2003) works as follows:
    # 
    # For each observation m:
    #   1. Compute the ensemble mean of the m-th observed variable: y_m_f
    #   2. Compute the ensemble anomalies of all state variables: a_i = x_i - x_mean
    #   3. Compute the regression: for each state variable i,
    #      beta_i = sum_n (a_i_n * a_m_n) / sum_n (a_m_n^2)  (where n indexes members)
    #      Apply localization: beta_i *= loc_weight(i, obs_index_m)
    #   4. Update: x_i_n += beta_i * (y_m - y_m_f) for all members n
    #   5. Recompute y_m_f from the updated ensemble (for the next observation)
    
    # This is equivalent to the Kalman update but done sequentially per observation.
    
    updated = inflated.copy()
    
    # Track the current ensemble mean for efficient updates
    current_mean = updated.mean(axis=0)
    
    for m, obs_idx in enumerate(observed_indices):
        # Check deadline
        if time.time() - t0 > _DEADLINE - 5:
            break
        
        # Current ensemble anomalies for the observed variable
        obs_anom = updated[:, obs_idx] - current_mean[obs_idx]  # (N,)
        
        # Variance of observed variable in ensemble
        var_obs = np.mean(obs_anom ** 2)
        
        if var_obs < 1e-15:
            continue
        
        # Innovation for this observation
        y_m = observations[m]
        y_m_f = current_mean[obs_idx]
        innovation = y_m - y_m_f
        
        # For each state variable, compute localized regression coefficient
        for i in range(dim):
            # Check deadline
            if time.time() - t0 > _DEADLINE - 5:
                break
            
            if i == obs_idx:
                # For the observed variable itself, the regression coefficient is 1.0
                # (we want to pull it toward the observation)
                # But with observation error, it should be less than 1.
                # In sequential EnSRF, the update for the observed variable is:
                # x_obs += (1 - R_m / (P_mm + R_m)) * innovation
                # where P_mm is the ensemble variance of the observed variable
                # and R_m is the observation variance.
                # The regression coefficient approach gives:
                # beta = P_mm / (P_mm + R_m) if we account for obs error
                # But in the simple sequential EnSRF without explicit R,
                # beta = 1 for the observed variable (full pull).
                # To incorporate observation error, we scale by:
                # beta_obs = P_mm / (P_mm + R_m)
                R_m = observation_variance
                beta = var_obs / (var_obs + R_m)
            else:
                # State variable anomalies
                state_anom = updated[:, i] - current_mean[i]  # (N,)
                
                # Regression coefficient
                cov_is = np.mean(state_anom * obs_anom)
                beta = cov_is / var_obs
                
                # Localization
                beta *= _localization_weight(i - obs_idx, dim, _loc_range)
            
            # Update all members for state variable i
            if abs(beta) > 1e-12:
                updated[:, i] += beta * innovation
        
        # Update current mean (only for the observed variable changed significantly)
        current_mean[obs_idx] = updated[:, obs_idx].mean()
        # For other variables, their means changed too, but we can recompute at the end
        # Actually, let's update all means to keep things consistent
        # But that's O(N*dim) per observation, which is 16*20*10 = 3200 ops, fine.
        current_mean = updated.mean(axis=0)
    
    # Recompute final mean
    final_mean = updated.mean(axis=0)
    
    # --- Step 3: Adaptive inflation update ---
    # Compare innovation norms to expected
    y_f_all = inflated[:, observed_indices]  # (N, n_obs)
    y_f_mean = y_f_all.mean(axis=0)  # (n_obs,)
    innovations_all = observations - y_f_mean  # (n_obs,)
    
    # Expected innovation variance: ensemble variance + obs variance
    P_obs_diag = np.array([np.var(inflated[:, obs_idx]) for obs_idx in observed_indices])
    expected_var = P_obs_diag + observation_variance
    expected_norm_sq = np.sum(expected_var)
    
    innovation_norm_sq = np.sum(innovations_all ** 2)
    
    if expected_norm_sq > 1e-15:
        ratio = np.sqrt(innovation_norm_sq / expected_norm_sq)
        target = min(max(ratio, 1.0), _inflation_max)
        _inflation = _inflation_alpha * _inflation + (1 - _inflation_alpha) * target
    else:
        _inflation = _inflation_alpha * _inflation + (1 - _inflation_alpha) * 1.0
    _inflation = np.clip(_inflation, _inflation_min, _inflation_max)
    
    # Safety: time guard
    if time.time() - t0 > _DEADLINE:
        updated = np.where(np.isfinite(updated), updated, 0.0)
        updated = np.clip(updated, -100, 100)
        return updated
    
    # Final sanitization
    updated = np.where(np.isfinite(updated), updated, 0.0)
    updated = np.clip(updated, -100, 100)
    
    return updated
