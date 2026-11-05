import numpy as np

def _rk4_step(x, F, dt=0.01):
    """One RK4 step for Lorenz-96 with periodic boundary conditions."""
    n = len(x)
    def f(y):
        dy = np.empty(n)
        for i in range(n):
            dy[i] = (y[(i+1) % n] - y[(i-2) % n]) * y[(i-1) % n] - y[i] + F
        return dy
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _propagate_ensemble(ensemble, F, n_steps=5, dt=0.01):
    """Propagate each ensemble member forward by n_steps RK4 steps."""
    result = ensemble.copy()
    for _ in range(n_steps):
        for i in range(result.shape[0]):
            result[i] = _rk4_step(result[i], F, dt)
    return result

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with localization and inflation.
    """
    n_members, n_vars = ensemble.shape
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # Estimate F from ensemble mean (Lorenz-96 fixed point is x_i = F)
    mean_ensemble = np.mean(ensemble, axis=0)
    F_est = float(np.mean(mean_ensemble))
    
    # Propagate ensemble forward by 5 RK4 steps
    propagated = _propagate_ensemble(ensemble, F_est, n_steps=5, dt=0.01)
    
    # Background mean and anomalies
    bg_mean = np.mean(propagated, axis=0)
    anomalies = propagated - bg_mean
    
    # Background covariance
    C = (anomalies.T @ anomalies) / (n_members - 1.0)
    
    # Observed block of covariance
    C_oo = C[np.ix_(obs_idx, obs_idx)]
    
    # Exponential localization
    loc_matrix = np.ones((n_obs, n_obs))
    for i in range(n_obs):
        for j in range(n_obs):
            dist = abs(int(obs_idx[i]) - int(obs_idx[j]))
            dist = min(dist, n_vars - dist)
            loc_matrix[i, j] = np.exp(-0.3 * dist)
    C_oo_loc = C_oo * loc_matrix
    
    # Observation noise covariance
    R = observation_variance * np.eye(n_obs)
    
    # Kalman gain
    S = C_oo_loc + R
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    K = C_oo_loc @ S_inv  # shape (n_obs, n_obs)
    
    # Innovation
    innovation = observations - bg_mean[obs_idx]
    
    # Update background mean
    bg_mean_updated = bg_mean.copy()
    bg_mean_updated[obs_idx] += K.T @ innovation
    
    # Update ensemble members
    for i in range(n_members):
        a_obs = anomalies[i, obs_idx]
        propagated[i, obs_idx] -= (K @ a_obs)
    
    # Inflation to prevent collapse
    spread_obs = np.std(propagated[:, obs_idx], axis=0)
    if np.max(spread_obs) > 1e-6:
        infl_factor = 1.0 + 0.01 * np.max(np.abs(innovation) / (spread_obs + 1e-8))
    else:
        infl_factor = 1.01
    infl_factor = min(infl_factor, 1.1)
    
    for i in range(n_members):
        propagated[i] = bg_mean_updated + infl_factor * (propagated[i] - bg_mean_updated)
    
    # Clip to valid range
    propagated = np.clip(propagated, -100.0, 100.0)
    
    return propagated
