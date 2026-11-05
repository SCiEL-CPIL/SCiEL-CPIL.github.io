"""
Ensemble Smoother with Multiple Scattering (ES-MSC) for Lorenz-96 assimilation.
Strategy:
1. Estimate the forcing parameter F from the ensemble statistics.
2. Treat the current state as the start of a 5-step RK4 window.
3. Iteratively (scattering steps):
   a. Propagate the ensemble forward through the dynamics (RK4, 5 steps).
   b. Apply a linear ensemble Kalman update (EnSRF) with localization and inflation.
   c. Re-initialize the ensemble to the updated state and repeat.
4. This captures non-linear error structure by re-propagating through the flow
   at each scattering iteration, correcting for bias inherent in single-step filters.
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
    """Propagate ensemble using RK4 for n_steps sub-steps."""
    x = x.copy()
    for _ in range(n_steps):
        k1 = _lorenz96_deriv(x, F)
        k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_deriv(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x

def _estimate_F(ensemble):
    """Estimate forcing F from ensemble mean assuming quasi-stationary balance."""
    mean_x = np.mean(ensemble, axis=0)
    n_vars = len(mean_x)
    F_estimates = np.zeros(n_vars)
    for i in range(n_vars):
        term = (mean_x[(i+1) % n_vars] - mean_x[(i-2) % n_vars]) * mean_x[(i-1) % n_vars] - mean_x[i]
        F_estimates[i] = -term
    return float(np.median(F_estimates))

def _localize_cov(cov, n_vars, L_loc=3.0):
    """Apply exponential localization to covariance matrix."""
    idx = np.arange(n_vars)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    mask = np.exp(-0.5 * (dist / L_loc) ** 2)
    return cov * mask

def _ensrf_update(X, observations, observed_indices, observation_variance, F, dt=0.01, n_prop_steps=5, n_scatter=5):
    """
    ES-MSC: iterative ensemble smoother with multiple scattering.
    X: (N, n_vars) ensemble at analysis time (background)
    observations: (n_obs,)
    observed_indices: (n_obs,)
    observation_variance: scalar
    Returns updated ensemble (N, n_vars).
    """
    N, n_vars = X.shape
    n_obs = len(observed_indices)
    
    # Build observation operator H as (n_obs, n_vars)
    H = np.zeros((n_obs, n_vars))
    for k, idx in enumerate(observed_indices):
        H[k, idx] = 1.0
    
    R = observation_variance * np.eye(n_obs)
    
    # Localization mask
    idx = np.arange(n_vars)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    mask = np.exp(-0.5 * (dist / 3.0) ** 2)
    
    # Initial ensemble
    X_curr = X.copy()
    
    for scatter in range(n_scatter):
        # Step 1: Propagate forward through dynamics
        X_prop = _rk4_step(X_curr, F, dt, n_prop_steps)
        
        # Step 2: Compute innovation
        H_X_T = H @ X_prop.T  # (n_obs, N)
        Innov = observations[:, None] - H_X_T  # (n_obs, N)
        
        # Step 3: Compute background error covariance (localized)
        mean_x = np.mean(X_prop, axis=0)
        X_c = X_prop - mean_x
        cov = (X_c.T @ X_c) / (N - 1)
        cov_loc = cov * mask
        
        # Step 4: Adaptive inflation
        S_diag = np.diag(H @ cov_loc @ H.T + R)
        innov_var = np.var(Innov, axis=1)
        ratio = np.mean(innov_var / np.maximum(S_diag, 1e-10))
        
        if ratio > 1.0:
            inflation = 1.0 + 0.5 * (ratio - 1.0)
        else:
            inflation = 1.0
        inflation = np.clip(inflation, 1.0, 1.5)
        
        # Apply inflation to perturbations
        X_c_inflated = X_c * np.sqrt(inflation)
        cov_inflated = (X_c_inflated.T @ X_c_inflated) / (N - 1)
        cov_loc_inflated = cov_inflated * mask
        
        # Step 5: Compute Kalman gain and update
        S = H @ cov_loc_inflated @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        K = cov_loc_inflated @ H.T @ S_inv  # (n_vars, n_obs)
        Update = K @ Innov  # (n_vars, N)
        
        X_new = X_prop + Update.T  # (N, n_vars)
        
        # Step 6: For next scattering, use the updated state as new background
        # But we need to re-propagate from this updated state
        X_curr = X_new
        
        # Check convergence
        if scatter > 0:
            if np.max(np.abs(X_new - X_curr)) < 1e-8:
                break
    
    return X_curr

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform ES-MSC assimilation update.
    ensemble: (16, 20)
    observations: (8,)
    observed_indices: (8,)
    observation_variance: scalar
    Returns updated ensemble (16, 20).
    """
    N, n_vars = ensemble.shape
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Estimate F
    F_est = _estimate_F(ensemble)
    
    # Check time before starting main loop
    if time.time() > deadline:
        return np.clip(np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0), -100.0, 100.0)
    
    # Perform ES-MSC update
    try:
        result = _ensrf_update(ensemble, observations, observed_indices, observation_variance, F_est)
    except Exception:
        # Fallback: simple EnKF update without propagation
        n_obs = len(observed_indices)
        H = np.zeros((n_obs, n_vars))
        for k, idx in enumerate(observed_indices):
            H[k, idx] = 1.0
        R = observation_variance * np.eye(n_obs)
        
        mean_x = np.mean(ensemble, axis=0)
        X_c = ensemble - mean_x
        cov = (X_c.T @ X_c) / (N - 1)
        
        idx = np.arange(n_vars)
        dist = np.abs(idx[:, None] - idx[None, :])
        dist = np.minimum(dist, n_vars - dist)
        mask = np.exp(-0.5 * (dist / 3.0) ** 2)
        cov_loc = cov * mask
        
        H_X_T = H @ ensemble.T
        Innov = observations[:, None] - H_X_T
        S = H @ cov_loc @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = cov_loc @ H.T @ S_inv
        result = ensemble + (K @ Innov).T
    
    # Safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
