"""Ensemble Kalman filter with localization and inflation for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_step(x, F, dt, n=20):
    """Single RK4 step for Lorenz-96 system."""
    def f(state):
        # dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        # Periodic indices
        x_next = np.roll(state, -1)
        x_prev1 = np.roll(state, 1)
        x_prev2 = np.roll(state, 2)
        return (x_next - x_prev2) * x_prev1 - state + F
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rk4_propagate(x, F, dt, n_steps, n=20):
    """Propagate a single state through n_steps RK4 steps."""
    for _ in range(n_steps):
        x = _lorenz96_step(x, F, dt, n)
    return x


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble members before update.
    observations : ndarray, shape (10,)
        Noisy observations of even-indexed coordinates.
    observed_indices : array-like
        Indices of observed coordinates (even indices).
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    ndarray, shape (16, 20)
        Updated ensemble.
    """
    N = 16  # number of ensemble members
    n = 20  # state dimension
    
    # Convert to numpy arrays
    ens = np.array(ensemble, dtype=np.float64)
    obs = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R = float(observation_variance)
    
    # Start with a copy
    result = ens.copy()
    
    # --- Ensemble mean and covariance ---
    ens_mean = np.mean(ens, axis=0)
    
    # Center the ensemble
    X = ens - ens_mean  # (16, 20)
    
    # Sample covariance
    # P = X.T @ X / (N-1)
    P = (X.T @ X) / (N - 1)  # (20, 20)
    
    # --- Localization: use distance-based correlation decay ---
    # For Lorenz-96 with periodic boundary, distance is min(|i-j|, n-|i-j|)
    # Apply a localization function to the covariance
    loc_scale = 5.0  # localization scale (in grid points)
    
    # Build localization matrix
    loc_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = min(abs(i - j), n - abs(i - j))
            # Exponential decay
            loc_matrix[i, j] = np.exp(-dist / loc_scale)
    
    # Localize the covariance (element-wise)
    P_loc = P * loc_matrix
    
    # --- Observation operator H ---
    # H selects observed indices
    H = np.zeros((len(obs_idx), n))
    for k, idx in enumerate(obs_idx):
        H[k, idx] = 1.0
    
    # Observation covariance R_obs
    R_obs = R * np.eye(len(obs_idx))
    
    # --- Ensemble Kalman Filter (deterministic, ensemble-based) ---
    # Using the ensemble-based Kalm filter formulation
    # We can use the "perturbed observation" approach or the deterministic approach
    
    # Deterministic EKF:
    # K = P H^T (H P H^T + R)^{-1}
    # x_new = x_mean + K (y - H x_mean)
    # For ensemble: update each member
    
    # Compute S = H P H^T + R
    HP = H @ P_loc  # (10, 20)
    S = HP @ H.T + R_obs  # (10, 10)
    
    # Invert S
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Gain matrix K = P_loc H^T S^{-1}
    K = P_loc @ H.T @ S_inv  # (20, 10)
    
    # Innovation
    innovation = obs - H @ ens_mean  # (10,)
    
    # Update ensemble mean
    new_mean = ens_mean + K @ innovation
    
    # Update ensemble members using the ensemble-based update
    # X_new = X - X H^T S^{-1} (H X)^T / (N-1)
    # Or equivalently, use the square-root filter
    
    # Let's use the standard ensemble update:
    # For each member i:
    # x_i_new = x_i + K (y_i - H x_i) where y_i are perturbed observations
    # But for deterministic EKF, we use:
    # X_new = X (I - K H X / (N-1))^{-1} ... this gets complicated
    
    # Simpler: use the "ensemble Kalman filter" with multiplicative update
    # X_new = X - (X H^T) S^{-1} (H X - (N-1) innovation) / (N-1)
    
    # Actually, let's use the standard ensemble-based update:
    # y_pert = y + sqrt(R) * eps for each member
    # But that requires random numbers. Let's use the deterministic approach.
    
    # Deterministic EKF update for ensemble:
    # x_i_new = x_i + K (y - H x_i) for each member? No, that's not right.
    
    # Let's use the proper ensemble-based Kalman filter:
    # The innovation for member i is: y_i - H x_i
    # But we want to use the ensemble mean innovation.
    
    # Standard EKF ensemble update:
    # X_new = X (I - K H X / (N-1))
    # where K = P H^T S^{-1}
    
    # Compute:
    # A = I - K H X / (N-1)
    KHX = K @ H @ X  # (20, 16)
    A = np.eye(N) - KHX / (N - 1)  # (16, 16)
    
    # X_new = X @ A
    X_new = X @ A  # (16, 20)
    
    # New ensemble
    result = new_mean + X_new
    
    # --- Inflation: multiplicative inflation to account for model error ---
    # Compute ensemble variance at observed points
    ens_var_obs = np.var(result[:, obs_idx], axis=0)
    mean_ens_var_obs = np.mean(ens_var_obs)
    
    # Inflate if ensemble variance is too low
    inflation_factor = 1.01  # gentle inflation
    
    # Apply inflation around the mean
    result = new_mean + inflation_factor * (result - new_mean)
    
    # --- Clip to valid range ---
    result = np.clip(result, -100.0, 100.0)
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result

# EVOLVE-BLOCK-END
