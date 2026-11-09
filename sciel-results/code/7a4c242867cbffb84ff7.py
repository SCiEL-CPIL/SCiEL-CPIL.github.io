"""Lorenz-96 ensemble assimilation with ensemble Kalman filter (EnKF) + inflation."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system with forcing F."""
    n = len(x)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
    return dx

def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5*dt*k1, F)
    k3 = _lorenz96_rhs(x + 0.5*dt*k2, F)
    k4 = _lorenz96_rhs(x + dt*k3, F)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def _localize_cov(cov, observed_indices, n, sigma):
    """Apply exponential localization to observation covariance matrix."""
    m = len(observed_indices)
    loc = np.ones((m, m))
    for i in range(m):
        for j in range(m):
            d = abs(observed_indices[i] - observed_indices[j])
            d = min(d, n - d)
            loc[i, j] = np.exp(-0.5 * (d / sigma) ** 2)
    return cov * loc

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter update with adaptive inflation and localization.
    
    Parameters:
        ensemble: (16, 20) array of ensemble members
        observations: (10,) array of noisy observations of even-indexed states
        observed_indices: indices of observed variables (even indices)
        observation_variance: scalar variance of observation noise
    
    Returns:
        Updated ensemble with same (16, 20) shape
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n_members, n_state = ensemble.shape
    n_obs = len(observed_indices)
    
    # Convert to float64 for numerical stability
    ens = ensemble.astype(np.float64)
    obs = observations.astype(np.float64)
    
    # Compute ensemble mean and perturbation matrix
    ens_mean = np.mean(ens, axis=0)
    ens_pert = ens - ens_mean  # (n_members, n_state)
    
    # Project to observed subspace
    ens_pert_obs = ens_pert[:, observed_indices]  # (n_members, n_obs)
    
    # Estimate ensemble covariance in observation space
    # Use (W-1) scaling for unbiased estimate
    scale = 1.0 / (n_members - 1)
    B = ens_pert_obs @ ens_pert_obs.T * scale  # (n_obs, n_obs)
    
    # Apply localization
    B = _localize_cov(B, observed_indices, n_state, sigma=3.0)
    
    # Observation error covariance
    R = observation_variance * np.eye(n_obs)
    
    # Innovation
    y = obs - ens_mean[observed_indices]  # (n_obs,)
    
    # Compute Kalman gain using Cholesky for numerical stability
    # S = B + R
    S = B + R
    
    # Add small jitter for Cholesky stability
    jitter = 1e-8 * np.trace(S) / n_obs
    S += jitter * np.eye(n_obs)
    
    try:
        chol_factor = cho_factor(S, lower=True, check_finite=False)
        # K = B @ S^{-1} = (B @ L^{-T}) @ L^{-1} where S = L @ L^T
        # Actually: K = B @ inv(S), and inv(S) = inv(L^T) @ inv(L)
        # B @ inv(S) = (B @ inv(L^T)) @ inv(L)
        # Solve: X @ L^T = B  =>  X = B @ L^{-T}
        # Then K = X @ L^{-1}
        BLt_inv = cho_solve(chol_factor, B, lower=True, check_finite=False)  # B @ L^{-T}
        K = BLt_inv @ np.linalg.inv(cho_factor(S, lower=True, check_finite=False)[0])
        # Actually let's do it more carefully:
        # S = L @ L^T
        # inv(S) = inv(L^T) @ inv(L)
        # K = B @ inv(S) = B @ inv(L^T) @ inv(L)
        # Let Y = B @ inv(L^T), solve Y @ L^T = B
        # Then K = Y @ inv(L), solve K @ L = Y
        L = cho_factor(S, lower=True, check_finite=False)[0]
        Y = np.linalg.solve(L.T, B.T).T  # Y = B @ L^{-T}
        K = np.linalg.solve(L, Y.T).T    # K = Y @ L^{-1}
    except Exception:
        # Fallback: use pseudo-inverse
        K = B @ np.linalg.pinv(S)
    
    # Innovation gain applied to all state variables
    # K_full: (n_state, n_obs)
    # We need to propagate the gain to unobserved variables via ensemble covariance
    # Full ensemble covariance
    P = ens_pert @ ens_pert.T * scale  # (n_members, n_members) -- wait, this is wrong
    # Actually P_state = ens_pert.T @ ens_pert * scale  # (n_state, n_state)
    P_state = ens_pert.T @ ens_pert * scale  # (n_state, n_state)
    
    # Cross-covariance between state and observations
    P_xo = ens_pert.T @ ens_pert_obs * scale  # (n_state, n_obs)
    
    # Full Kalman gain for all state variables
    K_full = P_xo @ np.linalg.pinv(S)  # (n_state, n_obs)
    
    # Apply localization to the gain for unobserved variables
    # The gain for observed variables should be strong; for unobserved, decay with distance
    gain_scale = np.ones(n_state)
    for i in range(n_state):
        min_dist = min(abs(i - oi) for oi in observed_indices)
        min_dist = min(min_dist, n_state - min_dist)
        gain_scale[i] = np.exp(-0.5 * (min_dist / 3.0) ** 2)
    
    K_full *= gain_scale[:, np.newaxis]
    
    # Update ensemble mean
    ens_mean_new = ens_mean + K_full @ y
    
    # Update ensemble perturbations (square-root form)
    # New perturbation: W' = W - K @ W_obs
    # where W_obs is the perturbation in observation space
    ens_pert_new = ens_pert - K_full @ ens_pert_obs
    
    # Adaptive inflation: adjust spread to match expected spread
    # Compute current spread vs expected
    # Simple adaptive inflation based on innovation consistency
    # Compute normalized innovation
    S_inv_y = np.linalg.solve(S, y)
    chi2 = y @ S_inv_y
    expected_chi2 = n_obs
    # Inflation factor
    if chi2 > expected_chi2 * 1.5:
        inflation = 1.0 + 0.05 * (chi2 / expected_chi2 - 1.0)
        inflation = min(inflation, 1.15)
    elif chi2 < expected_chi2 * 0.5:
        inflation = 1.0 - 0.03 * (1.0 - chi2 / expected_chi2)
        inflation = max(inflation, 0.95)
    else:
        inflation = 1.0
    
    # Apply inflation to perturbations
    ens_pert_new *= inflation
    
    # Reconstruct ensemble
    result = ens_mean_new[np.newaxis, :] + ens_pert_new
    
    # Clamp to valid range
    result = np.clip(result, -100.0, 100.0)
    
    # Ensure finite
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Time check
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Return best result found so far (which is just the current result)
        pass
    
    return result.astype(np.float64)

# EVOLVE-BLOCK-END
