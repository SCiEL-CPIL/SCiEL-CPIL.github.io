"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-computed constants
N_VARS = 20
N_MEMBERS = 16
F = 8.0  # We'll use F=8 as default but the function doesn't receive F directly
# Actually, F is not passed to analyze. We need to infer it or use a robust approach.
# The task says F=6,8,10. Since we don't know which one is being used at any given call,
# we need an approach that works well across all three.

def _lorenz96_step(x, F, dt=0.01):
    """Single RK4 step for Lorenz-96."""
    def f(x):
        return (np.roll(x, 1) - np.roll(x, 2)) * np.roll(x, -1) - x + F
    
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _lorenz96_propagate(x, F, n_steps=5, dt=0.01):
    """Propagate Lorenz-96 state for n_steps."""
    for _ in range(n_steps):
        x = _lorenz96_step(x, F, dt)
    return x

def _ring_distance(i, j, n=N_VARS):
    """Compute ring distance between indices i and j on a ring of size n."""
    d = abs(i - j)
    return min(d, n - d)

def _localization_kernel(dist, L=4.0):
    """Gaspari-Cohn style localization kernel."""
    # Simple exponential localization
    return np.exp(-0.5 * (dist / L) ** 2)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update using localized ensemble Kalman filter.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: indices that were observed (even indices 0,2,4,...,18)
    - observation_variance: variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    result = ensemble.copy()
    n_members, n_vars = result.shape
    n_obs = len(observed_indices)
    
    # Estimate F from the ensemble statistics
    # For Lorenz-96, the mean of x^2 relates to F
    # <x^2> ≈ F^2 / 3 for large N, but this is rough
    # Better: use the fact that for steady state, <(x_{i+1} - x_{i-2}) * x_{i-1} - x_i> ≈ 0
    # So <x_i> ≈ <(x_{i+1} - x_{i-2}) * x_{i-1}>
    # This is complex. Let's try to infer F from the variance.
    # For L96, variance ≈ F^2 / 3 roughly, so F ≈ sqrt(3 * var)
    var_estimate = np.var(result, axis=0).mean()
    F_est = np.sqrt(3.0 * max(var_estimate, 1.0))
    # Clamp F to reasonable range
    F_est = np.clip(F_est, 5.0, 12.0)
    
    # Ensemble mean and perturbation
    ens_mean = result.mean(axis=0)
    perturbation = result - ens_mean  # (n_members, n_vars)
    
    # Observation mean
    obs_mean = np.mean(observations)
    
    # Compute ensemble covariance at observed indices
    # P_oo = (1/(M-1)) * H * P * H^T where H picks observed components
    H = np.zeros((n_obs, n_vars))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Ensemble covariance (M-1 denominator for unbiased estimate)
    P = (perturbation.T @ perturbation) / (n_members - 1)  # (n_vars, n_vars)
    
    # Innovation covariance
    P_oo = H @ P @ H.T  # (n_obs, n_obs)
    R = observation_variance * np.eye(n_obs)
    S = P_oo + R  # (n_obs, n_obs)
    
    # Innovation
    innovation = observations - H @ ens_mean  # (n_obs,)
    
    # Gain: K = P H^T S^{-1}
    # We need P H^T which is (n_vars, n_obs)
    PHt = P @ H.T  # (n_vars, n_obs)
    
    # Solve S^{-1} * innovation
    try:
        S_inv_innov = np.linalg.solve(S, innovation)  # (n_obs,)
    except np.linalg.LinAlgError:
        S_inv_innov = np.linalg.pinv(S) @ innovation
    
    # Update: x_a = x_f + K * innovation
    update = PHt @ S_inv_innov  # (n_vars,)
    
    # Apply update to mean
    new_mean = ens_mean + update
    
    # Update perturbations using ensemble Kalman filter
    # For each member: x_a_i = x_f_i + K * (y_i - H x_f_i)
    # But we only have one observation vector, not per-member observations
    # Use the standard EnKF perturbation update
    
    # K = P H^T S^{-1} (n_vars, n_obs)
    K = PHt @ np.linalg.solve(S, np.eye(n_obs))  # (n_vars, n_obs)
    
    # Per-member innovation: y - H x_f_i
    # H x_f_i for each member
    Hx = perturbation @ H.T  # (n_members, n_obs)
    member_innov = observations - (ens_mean @ H.T + Hx)  # (n_members, n_obs)
    # Actually: y - H x_f_i = (y - H ens_mean) - H (x_f_i - ens_mean) = innovation - Hx
    member_innov = innovation[None, :] - Hx  # (n_members, n_obs)
    
    # Update each member's perturbation
    # x_a_i - new_mean = (x_f_i - ens_mean) + K * (y - H x_f_i) - (x_f_i - ens_mean + K * (y - H ens_mean))
    # = (x_f_i - ens_mean) - K * H (x_f_i - ens_mean)
    # = (I - K H) (x_f_i - ens_mean)
    
    KH = K @ H  # (n_vars, n_vars)
    A = np.eye(n_vars) - KH  # (n_vars, n_vars)
    
    new_perturbation = A @ perturbation  # (n_vars, n_members) -> need to be careful with orientation
    # perturbation is (n_members, n_vars), so A @ perturbation.T gives (n_vars, n_members)
    new_perturbation = (A @ perturbation.T).T  # (n_members, n_vars)
    
    # Apply inflation to prevent filter collapse
    # Estimate innovation variance and compare to expected
    # If innovation variance is too small, inflate
    innov_var = np.var(innovation)
    expected_innov_var = np.trace(S) / n_obs
    inflation_factor = 1.0
    if expected_innov_var > 0:
        ratio = innov_var / expected_innov_var
        if ratio < 0.8:
            inflation_factor = 1.0 + 0.1 * (0.8 - ratio)
        elif ratio > 1.2:
            inflation_factor = 1.0 - 0.05 * (ratio - 1.2)
    inflation_factor = np.clip(inflation_factor, 0.95, 1.15)
    
    new_perturbation *= inflation_factor
    
    # Reconstruct ensemble
    result = new_mean[None, :] + new_perturbation
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, ens_mean[None, :])
    
    # Clamp to [-100, 100]
    result = np.clip(result, -100.0, 100.0)
    
    return result
# EVOLVE-BLOCK-END
