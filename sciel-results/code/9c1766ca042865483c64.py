import numpy as np
import time

def _rk4_step(state, F, dt):
    """Single RK4 step for Lorenz-96."""
    n = len(state)
    
    def lorenz96(x):
        return (np.roll(x, 1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    
    k1 = lorenz96(state)
    k2 = lorenz96(state + 0.5 * dt * k1)
    k3 = lorenz96(state + 0.5 * dt * k2)
    k4 = lorenz96(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _propagate_ensemble(ensemble, F, n_steps=5, dt=0.01):
    """Propagate ensemble members through RK4 steps."""
    n_members = ensemble.shape[0]
    result = ensemble.copy()
    for m in range(n_members):
        for _ in range(n_steps):
            result[m] = _rk4_step(result[m], F, dt)
    return result

def _estimate_F(ensemble, observations, observed_indices):
    """Estimate forcing parameter F from ensemble statistics."""
    # We can estimate F from the equilibrium: in Lorenz-96, the average of x_i^2 ≈ F^2/2
    # More robust: use the relation that at equilibrium, <(x_{i+1}-x_{i-2})*x_{i-1}> = <x_i> + <x_i>... 
    # Actually, taking expectation of dx_i/dt = 0: <(x_{i+1}-x_{i-2})*x_{i-1}> = <x_i> + F
    # So F = <(x_{i+1}-x_{i-2})*x_{i-1}> - <x_i>
    n = ensemble.shape[1]
    F_estimates = []
    for m in range(ensemble.shape[0]):
        x = ensemble[m]
        for i in range(n):
            i1 = (i+1) % n
            i2 = (i-2) % n
            i1m = (i-1) % n
            F_est = (x[i1] - x[i2]) * x[i1m] - x[i]
            F_estimates.append(F_est)
    return np.mean(F_estimates)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter analysis update for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (16,) or (16, 10) array of observations
    - observed_indices: indices of observed variables
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same shape (16, 20)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n_members, n_vars = ensemble.shape
    obs_indices = np.array(observed_indices)
    n_obs = len(obs_indices)
    
    # Handle different observation shapes
    if observations.ndim == 1:
        obs = observations.reshape(1, -1)
        # If single observation vector, broadcast
        if len(obs[0]) == n_obs:
            obs = np.tile(obs, (n_members, 1))
        else:
            obs = np.tile(observations.reshape(1, n_obs), (n_members, 1))
    else:
        obs = observations
    
    # Ensure observation variance is scalar
    if np.isscalar(observation_variance):
        R = observation_variance * np.eye(n_obs)
    else:
        R = np.array(observation_variance)
    
    # Estimate F from ensemble (use median for robustness)
    F_est = _estimate_F(ensemble, observations, observed_indices)
    # Clamp F to reasonable range
    F_est = np.clip(F_est, 1.0, 20.0)
    
    # Propagate ensemble forward 5 RK4 steps
    ensemble_prop = _propagate_ensemble(ensemble, F_est, n_steps=5, dt=0.01)
    
    # Check time
    if time.time() > deadline - 5.0:
        return np.clip(ensemble_prop, -100, 100)
    
    # Compute ensemble mean and covariance
    ens_mean = np.mean(ensemble_prop, axis=0)
    ens_anoms = ensemble_prop - ens_mean
    
    # Sample covariance (with bias correction)
    S = (ens_anoms.T @ ens_anoms) / (n_members - 1)
    
    # Localization: use a simple exponential decay based on index distance
    # For Lorenz-96, correlations decay over a few indices
    loc_scale = 4.0  # Localization scale in index space
    loc_matrix = np.ones((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            dist = min(abs(i - j), n_vars - abs(i - j))
            loc_matrix[i, j] = np.exp(-dist / loc_scale)
    
    # Localize the covariance
    S_loc = S * loc_matrix
    
    # Observation operator: select observed indices
    H = np.zeros((n_obs, n_vars))
    for k, idx in enumerate(obs_indices):
        H[k, int(idx)] = 1.0
    
    # Observation covariance R (diagonal)
    R = observation_variance * np.eye(n_obs)
    
    # Innovation: y - H*x_bar
    Hx = H @ ens_mean
    innovation = obs[:, 0] - Hx  # Use first member's obs if shape is (n_members, n_obs)
    
    # Actually, obs might be per-member. Let's handle both cases.
    # If obs is (n_members, n_obs), use the mean observation
    if obs.shape[0] == n_members:
        obs_mean = np.mean(obs, axis=0)
    else:
        obs_mean = obs[0] if obs.shape[0] == 1 else obs
    
    innovation = obs_mean - Hx
    
    # Kalman gain computation
    # P_yy = H S H^T + R
    H_S_Ht = H @ S_loc @ H.T
    P_yy = H_S_Ht + R
    
    # Invert P_yy (small matrix, n_obs x n_obs)
    try:
        P_yy_inv = np.linalg.inv(P_yy)
    except np.linalg.LinAlgError:
        P_yy_inv = np.linalg.pinv(P_yy)
    
    # Kalman gain: K = S H^T P_yy^{-1}
    S_Ht = S_loc @ H.T
    K = S_Ht @ P_yy_inv
    
    # Update ensemble mean
    ens_mean_updated = ens_mean + K @ innovation
    
    # Update each ensemble member
    updated_ensemble = np.zeros_like(ensemble_prop)
    for m in range(n_members):
        updated_ensemble[m] = ensemble_prop[m] + K @ (obs[m] - Hx) if obs.shape[0] == n_members else \
                              ensemble_prop[m] + K @ innovation
    
    # Inflation: multiplicative inflation to prevent ensemble collapse
    # Compute spread before and after, inflate if needed
    spread_before = np.mean(np.var(ensemble_prop, axis=0))
    spread_after = np.mean(np.var(updated_ensemble, axis=0))
    
    # Adaptive inflation
    if spread_after > 0:
        inflation_factor = np.sqrt(spread_before / spread_after)
        inflation_factor = np.clip(inflation_factor, 1.0, 1.5)
    else:
        inflation_factor = 1.0
    
    # Apply inflation around the updated mean
    updated_mean = np.mean(updated_ensemble, axis=0)
    updated_ensemble = updated_mean + inflation_factor * (updated_ensemble - updated_mean)
    
    # Clip to valid range
    updated_ensemble = np.clip(updated_ensemble, -100, 100)
    
    # Ensure finite values
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
