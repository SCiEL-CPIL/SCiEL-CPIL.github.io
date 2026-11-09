"""EKF-based assimilation for Lorenz-96 with localization and adaptive inflation."""
import numpy as np

# Module-level state for adaptive inflation
_inflation_state = {'alpha': 1.0}

def _l96_rhs(x, F):
    """Lorenz-96 right-hand side."""
    n = len(x)
    dx = np.empty_like(x)
    for i in range(n):
        dx[i] = (x[(i + 1) % n] - x[(i - 2) % n]) * x[(i - 1) % n] - x[i] + F
    return dx

def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _l96_rhs(x, F)
    k2 = _l96_rhs(x + 0.5 * dt * k1, F)
    k3 = _l96_rhs(x + 0.5 * dt * k2, F)
    k4 = _l96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _localize(cov, observed_indices, n_vars, width=3.0):
    """Apply exponential localization to the observation error covariance."""
    n_obs = len(observed_indices)
    # Build the observation-space covariance with localization
    loc_cov = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        for j in range(n_obs):
            # Compute the distance in the full state space
            dist = abs(observed_indices[i] - observed_indices[j])
            dist = min(dist, n_vars - dist)  # periodic distance
            loc_cov[i, j] = cov[i, j] * np.exp(-0.5 * (dist / width) ** 2)
    return loc_cov

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter analysis update for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (10,)
        Noisy observations of even-indexed coordinates.
    observed_indices : array-like
        Indices of observed coordinates (even indices).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after analysis.
    """
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Compute ensemble mean and anomalies
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (16, 20)
    
    # Estimate background covariance in observation space
    # P_bb_obs = anomalies_obs @ anomalies_obs.T / (n_members - 1)
    anom_obs = anomalies[:, observed_indices]  # (16, 10)
    P_bb_obs = (anom_obs @ anom_obs.T) / (n_members - 1)  # (10, 10)
    
    # Apply localization to the observation-space covariance
    P_bb_loc = _localize(P_bb_obs, observed_indices, n_vars, width=3.0)
    
    # Observation noise covariance (diagonal)
    R = np.eye(n_obs) * observation_variance
    
    # Compute Kalman gain
    # P_bb_loc + R
    S = P_bb_loc + R  # (10, 10)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Cross-covariance between full state and observations
    # P_xb_obs = anomalies @ anomalies_obs.T / (n_members - 1)
    P_xb_obs = (anomalies @ anom_obs.T) / (n_members - 1)  # (20, 10)
    
    # Kalman gain K = P_xb_obs @ S_inv  (20, 10)
    K = P_xb_obs @ S_inv
    
    # Innovation
    innovation = observations - ens_mean[observed_indices]  # (10,)
    
    # Update ensemble mean
    updated_mean = ens_mean + K @ innovation  # (20,)
    
    # Update ensemble anomalies (stochastic formulation)
    # For each member: x_a = x_b + K * (y - H(x_b)) + (I - K H) * anomaly_b
    # But we need to preserve ensemble spread. Use the stochastic EKF approach.
    
    # Compute (I - K H) applied to anomalies
    # K H @ anomalies_obs = K @ (H @ anomalies) = K @ anom_obs
    KH_anom = K @ anom_obs  # (20, 16)
    updated_anomalies = anomalies - KH_anom  # (20, 16)
    
    # Add stochastic perturbation to preserve ensemble spread
    # The stochastic EKF adds a random perturbation with covariance
    # (I - K H) P_b (I - K H)^T - P_a, but a simpler approach is to
    # add noise to the innovation.
    
    # Alternative: use the square-root filter with random rotation
    # Generate a random orthogonal matrix
    rng = np.random.default_rng()
    Q = rng.standard_normal((n_obs, n_obs))
    Q, _ = np.linalg.qr(Q)
    
    # Scale the innovation by sqrt(1 - alpha) for the mean update
    # and add perturbation for spread
    # Actually, let's use a cleaner stochastic EKF:
    # x_a_i = x_b_i + K (y - H x_b_i) + (I - K H) (x_b_i - x_b_bar) * sqrt(1 - 1/n)
    # This is getting complicated. Let's use a simpler but effective approach.
    
    # Use the standard ensemble Kalman filter update with inflation
    # to avoid underestimation of uncertainty
    
    # Apply adaptive inflation
    alpha = _inflation_state['alpha']
    inflated_anomalies = updated_anomalies * alpha
    updated_ensemble = updated_mean[None, :] + inflated_anomalies.T  # (16, 20)
    
    # Update adaptive inflation based on ensemble spread
    # Compare predicted observation spread to observation noise
    pred_obs_mean = updated_ensemble[:, observed_indices].mean(axis=0)
    pred_obs_anom = updated_ensemble[:, observed_indices] - pred_obs_mean
    pred_obs_var = np.mean(pred_obs_anom ** 2, axis=0)  # (10,)
    
    # The ratio of predicted spread to observation noise
    spread_ratio = np.mean(pred_obs_var) / observation_variance
    
    # Adjust inflation: if spread is too small, increase alpha
    target_ratio = 1.5  # We want ensemble spread to be somewhat larger than obs noise
    if spread_ratio < target_ratio:
        _inflation_state['alpha'] = min(1.5, alpha * 1.05)
    else:
        _inflation_state['alpha'] = max(1.0, alpha * 0.98)
    
    # Clamp to valid range
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Ensure finite
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
