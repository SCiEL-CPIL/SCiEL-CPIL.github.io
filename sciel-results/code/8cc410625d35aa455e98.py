"""
Ensemble Smoother with Multiple Scattering (ES-MSC) for Lorenz-96 assimilation.
Strategy:
1. Treat the input ensemble as the background at the start of a 5-step forecast window.
2. Iteratively (scattering steps):
   a. Propagate the ensemble forward through the dynamics (RK4, 5 steps).
   b. Apply a linear ensemble Kalman update (EnSRF) with localization and inflation.
3. The final analysis is the ensemble at the end of the window after the last scattering step.
This preserves non-Gaussian structure better than a single linear update and accounts for
dynamics over the full assimilation window.
"""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96 system. x is (N, 20)."""
    x_1 = np.roll(x, 1, axis=1)
    x_m1 = np.roll(x, -1, axis=1)
    x_m2 = np.roll(x, 2, axis=1)
    return (x_1 - x_m2) * x_m1 - x + F

def _rk4_step(x, F, dt, n_steps=1):
    """Propagate ensemble using RK4."""
    x = x.copy()
    for _ in range(n_steps):
        k1 = _lorenz96_deriv(x, F)
        k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_deriv(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x

def _estimate_F(ensemble):
    """Estimate the forcing parameter F from the ensemble statistics."""
    N, n_vars = ensemble.shape
    mean_x = np.mean(ensemble, axis=0)
    F_estimates = np.zeros(n_vars)
    for i in range(n_vars):
        term = (mean_x[(i+1)%n_vars] - mean_x[(i-2)%n_vars]) * mean_x[(i-1)%n_vars] - mean_x[i]
        F_estimates[i] = -term
    return np.median(F_estimates)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform ES-MSC update.
    ensemble: (16, 20)
    observations: (8,)
    observed_indices: (8,)
    observation_variance: scalar
    """
    N, n_vars = ensemble.shape
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Estimate F
    F_est = _estimate_F(ensemble)
    
    # Setup observation operator
    n_obs = len(observed_indices)
    H = np.zeros((n_obs, n_vars))
    for k, idx_obs in enumerate(observed_indices):
        H[k, idx_obs] = 1.0
        
    R = observation_variance * np.eye(n_obs)
    
    # Localization mask
    L_loc = 3.0
    idx = np.arange(n_vars)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    mask = np.exp(-0.5 * (dist / L_loc) ** 2)
    
    # Initialize with the background
    X = ensemble.copy()
    
    # Number of scattering iterations
    n_scatter = 6
    
    for sc in range(n_scatter):
        # Check time
        if time.time() > deadline:
            break
            
        # a. Propagate forward through the dynamics (5 steps of dt=0.01)
        X_pred = _rk4_step(X, F_est, dt=0.01, n_steps=5)
        
        # b. Apply EnSRF update
        mean_x = np.mean(X_pred, axis=0)
        X_c = X_pred - mean_x
        cov = (X_c.T @ X_c) / (N - 1)
        cov_loc = cov * mask
        
        # Observation innovation
        H_X_T = H @ X_pred.T  # (n_obs, N)
        Innov = observations[:, None] - H_X_T  # (n_obs, N)
        
        # Adaptive inflation
        innov_var = np.var(Innov, axis=1)
        S_diag = np.diag(H @ cov_loc @ H.T + R)
        ratio = np.mean(innov_var / S_diag)
        
        if ratio > 1.0:
            inflation = 1.0 + 0.3 * (ratio - 1.0)
        else:
            inflation = 1.0
        inflation = np.clip(inflation, 1.0, 1.5)
        
        # Apply inflation
        X_c_inflated = X_c * np.sqrt(inflation)
        cov_inflated = (X_c_inflated.T @ X_c_inflated) / (N - 1)
        cov_loc_inflated = cov_inflated * mask
        
        # Compute Kalman Gain
        S = H @ cov_loc_inflated @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            
        K = cov_loc_inflated @ H.T @ S_inv
        
        # Update
        Update = K @ Innov
        X_new = X_pred + Update.T
        
        # Check convergence
        if np.max(np.abs(X_new - X_pred)) < 1e-8:
            X = X_new
            break
            
        X = X_new
    
    result = X
    
    # Safety Checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
