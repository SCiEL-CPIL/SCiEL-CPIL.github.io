"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute RHS of Lorenz-96 system for a single state vector."""
    n = len(x)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x[(i + 1) % n] - x[(i - 2) % n]) * x[(i - 1) % n] - x[i] + F
    return dx

def _lorenz96_rhs_batch(X, F):
    """Compute RHS for a batch of state vectors, shape (N, 20)."""
    N = X.shape[0]
    n = X.shape[1]
    dX = np.empty_like(X)
    # Use rolling for efficiency
    xp1 = np.roll(X, -1, axis=1)
    xm2 = np.roll(X, 2, axis=1)
    xm1 = np.roll(X, 1, axis=1)
    dX = (xp1 - xm2) * xm1 - X + F
    return dX

def _rk4_step(X, dt, F, steps=1):
    """Propagate ensemble X (shape (N, 20)) forward by 'steps' RK4 steps of size dt."""
    for _ in range(steps):
        k1 = _lorenz96_rhs_batch(X, F)
        k2 = _lorenz96_rhs_batch(X + 0.5 * dt * k1, F)
        k3 = _lorenz96_rhs_batch(X + 0.5 * dt * k2, F)
        k4 = _lorenz96_rhs_batch(X + dt * k3, F)
        X = X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return X

def _localize_cov(cov, obs_idx, state_dim, localization_scale=2.5):
    """Apply exponential localization to observation covariance matrix."""
    # cov is (m, m) where m = len(obs_idx)
    m = len(obs_idx)
    # Distance matrix between observed indices
    dist = np.abs(obs_idx[:, None] - obs_idx[None, :])
    # Periodic distance
    dist = np.minimum(dist, state_dim - dist)
    # Exponential localization
    weights = np.exp(-0.5 * (dist / localization_scale) ** 2)
    # Apply to covariance
    cov_loc = cov * weights
    return cov_loc

def _ensrf_update(ensemble, observations, observed_indices, obs_var, 
                  localization_scale=2.5, inflation_bounds=(0.7, 1.5)):
    """
    Perform a single linear Ensemble Square Root Filter update.
    
    ensemble: (N, D)
    observations: (N, m) - one observation per member for each observed index
    observed_indices: array of indices
    obs_var: float
    """
    N, D = ensemble.shape
    m = len(observed_indices)
    
    # Compute ensemble mean
    x_mean = np.mean(ensemble, axis=0)
    
    # Compute ensemble anomalies
    anomalies = ensemble - x_mean  # (N, D)
    
    # Observation anomalies
    obs_anom = observations - np.mean(observations, axis=0)  # (N, m)
    
    # Cross-covariance between state and observations
    # S = (1/(N-1)) * anomalies[:, obs_idx].T @ obs_anom  (D_obs, m)
    state_obs_anom = anomalies[:, observed_indices]  # (N, m)
    S = (state_obs_anom.T @ obs_anom) / (N - 1)  # (m, m)
    
    # Observation covariance
    R = obs_var * np.eye(m)
    
    # Localize the cross-covariance / background covariance
    # Actually, we localize the background covariance in observation space
    B_obs = (state_obs_anom.T @ state_obs_anom) / (N - 1)  # (m, m)
    B_obs_loc = _localize_cov(B_obs, observed_indices, D, localization_scale)
    
    # Innovation covariance
    P_yy = B_obs_loc + R  # (m, m)
    
    # Kalman gain: K = S @ inv(P_yy)  (m, m)
    # We need K such that x_a = x_b + K @ (y - H x_b)
    # For observation operator H = I (subset), the gain in observation space is:
    # K_obs = S @ inv(P_yy)
    try:
        K_obs = S @ np.linalg.inv(P_yy)  # (m, m)
    except np.linalg.LinAlgError:
        K_obs = np.eye(m) * 0.5
    
    # Update observed components
    # y_anom = observations - H @ x_mean
    y_anom = observations - x_mean[observed_indices][None, :]  # (N, m)
    
    # Delta in observation space: d_x_obs = K_obs @ y_anom.T  (m, N)
    d_x_obs = (K_obs @ y_anom.T).T  # (N, m)
    
    # For unobserved components, we use the cross-covariance
    # Full state cross-cov: C = anomalies.T @ obs_anom / (N-1)  (D, m)
    C = (anomalies.T @ obs_anom) / (N - 1)  # (D, m)
    
    # Localize C in the state dimension? Actually localization is usually on the observation space.
    # But for the gain, we compute K = C @ inv(P_yy)
    # However, we already localized B_obs. Let's be consistent.
    # Standard EnSRF: K = C @ (C^T C + R)^{-1} where C is cross-cov (D, m)
    # With localization, we modify the effective covariance.
    
    # Let's use the standard approach:
    # K = C @ inv(P_yy_loc) where P_yy_loc is localized
    # But C itself might need localization in the state dimension for spatial spread.
    # For now, apply localization to the observation covariance part.
    
    # Actually, a simpler and more robust approach for small N:
    # Use the localized B_obs to compute the gain for observed variables,
    # and for unobserved variables, use the correlation with observed variables.
    
    # Let's compute the full gain K (D, m)
    # K = C @ inv(P_yy)
    # But we want to apply localization. A common approach is to localize the background covariance.
    
    # Let's stick with the observation-space localization and compute gain for all states:
    # K = C @ inv(P_yy_loc)
    try:
        K = C @ np.linalg.inv(P_yy)  # (D, m)
    except np.linalg.LinAlgError:
        K = np.zeros((D, m))
    
    # Update all state components
    d_x = y_anom @ K.T  # (N, D)
    
    ensemble_new = ensemble + d_x
    
    # Adaptive inflation
    # Check ensemble spread vs observation error
    if N > 2:
        # Spread in observation space
        spread_obs = np.std(ensemble_new[:, observed_indices], axis=0)
        obs_spread = np.std(observations, axis=0)
        
        # Inflation factor
        # Target: spread should be comparable to obs error
        ratio = np.mean(spread_obs) / np.sqrt(obs_var)
        
        if ratio < 1.0:
            # Inflate
            inflation = 1.0 + 0.1 * (1.0 - ratio)
        elif ratio > 2.0:
            # Deflate
            inflation = 1.0 - 0.05 * (ratio - 2.0)
        else:
            inflation = 1.0
        
        # Clip inflation
        inflation = np.clip(inflation, inflation_bounds[0], inflation_bounds[1])
        
        # Apply inflation around the mean
        new_mean = np.mean(ensemble_new, axis=0)
        ensemble_new = new_mean + inflation * (ensemble_new - new_mean)
    
    return ensemble_new

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Non-linear Ensemble Smoother with Multiple Scattering (ES-MSC).
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (16, 10)
        Noisy observations of observed indices for each ensemble member.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    ensemble = np.array(ensemble, dtype=float)
    observations = np.array(observations, dtype=float)
    observed_indices = np.array(observed_indices)
    
    N, D = ensemble.shape
    m = len(observed_indices)
    
    # We need to infer F. From the task, F is 6, 8, or 10.
    # We can estimate F from the ensemble statistics or use a default.
    # The mean of x_i should be around F in steady state for L96.
    # Let's estimate F from the ensemble mean
    F_est = np.mean(ensemble, axis=0).mean()
    # Round to nearest integer (F is 6, 8, or 10)
    F = round(F_est)
    if F not in [6, 8, 10]:
        F = 8  # default
    
    # ES-MSC: iterate the linear update with re-propagation
    # Since we don't have the full propagation context (dt, number of steps),
    # and the evaluator propagates between calls, we can only do the analysis update.
    # However, the directive says to "re-propagate the ensemble through the Lorenz-96 dynamics at each scattering step".
    # But we don't know dt or how many steps. The evaluator does 5 steps of dt=0.01 between updates.
    # So we can propagate forward by 5 steps of dt=0.01 after each linear update to simulate the non-linear effect.
    
    dt = 0.01
    n_steps = 5
    
    max_scatterings = 3
    current_ensemble = ensemble.copy()
    
    for scattering in range(max_scatterings):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Linear EnSRF update
        current_ensemble = _ensrf_update(
            current_ensemble, observations, observed_indices, observation_variance,
            localization_scale=2.5, inflation_bounds=(0.7, 1.5)
        )
        
        # Re-propagate to capture non-linear dynamics
        # This simulates the effect of the flow on the updated ensemble
        if scattering < max_scatterings - 1:
            current_ensemble = _rk4_step(current_ensemble, dt, F, steps=n_steps)
    
    # Final check for finite values and bounds
    if not np.all(np.isfinite(current_ensemble)):
        current_ensemble = np.nan_to_num(current_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Clip to [-100, 100]
    current_ensemble = np.clip(current_ensemble, -100.0, 100.0)
    
    return current_ensemble

# EVOLVE-BLOCK-END
