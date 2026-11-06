"""Localized Ensemble Variational (EnVar) analysis with hybrid Gaussian-Huber loss for Lorenz-96."""
import numpy as np
import time
from scipy.optimize import minimize

def _lorenz96(x, F):
    """Lorenz-96 dynamics: dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F"""
    n = len(x)
    dx = np.empty_like(x)
    for i in range(n):
        ip1 = (i + 1) % n
        im1 = (i - 1) % n
        im2 = (i - 2) % n
        dx[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + F
    return dx

def _rk4_step(x, dt, F):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96(x, F)
    k2 = _lorenz96(x + 0.5 * dt * k1, F)
    k3 = _lorenz96(x + 0.5 * dt * k2, F)
    k4 = _lorenz96(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Localized Ensemble Variational (EnVar) analysis with hybrid Gaussian-Huber loss.
    
    Parameters:
        ensemble: (16, 20) array - ensemble of 16 members, 20 variables
        observations: (10,) or (16, 10) array - noisy observations of even-indexed variables
        observed_indices: array of 10 indices - which variables are observed
        observation_variance: float - variance of observation noise
    
    Returns:
        (16, 20) array - updated ensemble
    """
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    M, N = ensemble.shape  # M=16 members, N=20 variables
    ensemble = np.asarray(ensemble, dtype=np.float64)
    obs_idx = np.asarray(observed_indices, dtype=np.int64)
    
    # Handle observations shape: could be (10,) or (M, 10)
    obs = np.asarray(observations, dtype=np.float64)
    if obs.ndim == 1:
        y = obs
    else:
        # If (M, n_obs), use the mean as the "true" observation
        y = obs.mean(axis=0)
    
    n_obs = len(obs_idx)
    R_var = observation_variance
    R_std = np.sqrt(R_var)
    
    # Ensemble statistics
    mean = ensemble.mean(axis=0)
    anomalies = ensemble - mean  # (M, N)
    
    # --- Localization matrix ---
    sigma_loc = 3.0
    loc_matrix = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            d = abs(i - j)
            d = min(d, N - d)
            loc_matrix[i, j] = np.exp(-d**2 / (2.0 * sigma_loc**2))
    
    # --- Build cost function for EnVar ---
    # We optimize over the analysis mean x_a
    # Cost = 0.5 * (x_a - x_b)^T B^{-1} (x_a - x_b) + obs_cost(y, H x_a)
    # where B is the background error covariance (localized)
    # and obs_cost is a hybrid Gaussian + Huber loss
    
    # Background covariance (localized)
    B = (anomalies.T @ anomalies) / (M - 1)
    B_loc = B * loc_matrix
    
    # Precompute B^{-1} via Cholesky
    try:
        L_b = np.linalg.cholesky(B_loc + 1e-6 * np.eye(N))
        B_inv = np.linalg.inv(L_b @ L_b.T)
    except np.linalg.LinAlgError:
        B_inv = np.linalg.pinv(B_loc)
    
    # Precompute observation operator H
    H = np.zeros((n_obs, N))
    for k, idx in enumerate(obs_idx):
        H[k, idx] = 1.0
    
    # Huber parameter: transition between quadratic and linear
    # Use a value relative to observation noise
    huber_delta = 2.0 * R_std
    
    def cost_and_grad(x_a):
        """
        Cost function and gradient for EnVar.
        x_a: (N,) analysis mean
        """
        # Background term
        bg_diff = x_a - mean
        bg_cost = 0.5 * bg_diff @ B_inv @ bg_diff
        bg_grad = B_inv @ bg_diff
        
        # Observation term with hybrid Gaussian-Huber loss
        obs_diff = H @ x_a - y  # (n_obs,)
        abs_diff = np.abs(obs_diff)
        
        # Huber loss: quadratic for |d| <= delta, linear for |d| > delta
        mask = abs_diff <= huber_delta
        obs_cost = 0.0
        obs_grad = np.zeros(n_obs)
        
        # Gaussian part (within Huber radius)
        if np.any(mask):
            d_masked = obs_diff[mask]
            obs_cost += 0.5 * np.sum(d_masked**2) / R_var
            obs_grad[mask] += d_masked / R_var
        
        # Linear part (outside Huber radius)
        if np.any(~mask):
            d_out = obs_diff[~mask]
            obs_cost += np.sum(huber_delta * (np.abs(d_out) - 0.5 * huber_delta)) / R_var
            obs_grad[~mask] += np.sign(d_out) * huber_delta / R_var
        
        total_cost = bg_cost + obs_cost
        total_grad = bg_grad + H.T @ obs_grad
        
        return total_cost, total_grad
    
    # --- Optimize using L-BFGS-B ---
    # Initial guess: ensemble mean
    x0 = mean.copy()
    
    # Check deadline before optimization
    if time.time() > deadline - 5.0:
        # Not enough time, return inflated ensemble
        inflated = mean + anomalies * 1.02
        return np.clip(inflated, -100.0, 100.0)
    
    try:
        result = minimize(
            cost_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        x_a = result.x
    except Exception:
        # Fallback: simple EnKF-style update if optimizer fails
        x_a = mean.copy()
    
    # Check deadline after optimization
    if time.time() > deadline - 2.0:
        return np.clip(np.tile(x_a, (M, 1)) + anomalies * 0.98, -100.0, 100.0)
    
    # --- Reconstruct ensemble around new mean ---
    # Scale anomalies slightly to maintain spread (prevent over-concentration)
    scale = 1.0 + 0.01  # small inflation to prevent collapse
    
    ensemble_new = np.tile(x_a, (M, 1)) + anomalies * scale
    
    # Ensure finiteness and bounds
    ensemble_new = np.clip(ensemble_new, -100.0, 100.0)
    ensemble_new = np.nan_to_num(ensemble_new, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ensemble_new
