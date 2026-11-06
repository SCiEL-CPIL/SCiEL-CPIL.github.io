"""Ensemble Kalman Inversion (EnKI) with adaptive regularization for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_start_time = None
_adaptive_alpha = 1.0  # adaptive Tikhonov regularization parameter

def _get_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Inversion (EnKI) analysis update.
    
    Treats the unobserved variables as control variables and minimizes a
    misfit cost function with Tikhonov regularization to the background ensemble.
    Uses multiple gradient descent steps with adaptive learning rate.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar observation noise variance
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    M, N = ensemble.shape
    obs_var = float(observation_variance)
    obs_idx = np.asarray(observed_indices, dtype=int)
    L = len(obs_idx)
    
    # Deadline guard
    elapsed = _get_time()
    if elapsed > 140.0:
        result = np.array(ensemble, dtype=np.float64).copy()
        obs_arr = np.asarray(observations, dtype=np.float64)
        if obs_arr.ndim == 1:
            obs_arr = np.tile(obs_arr, (M, 1))
        result[:, obs_idx] += 0.5 * (obs_arr - result[:, obs_idx])
        np.clip(result, -100.0, 100.0, out=result)
        return result
    
    X_b = np.array(ensemble, dtype=np.float64)
    y_obs = np.asarray(observations, dtype=np.float64)
    if y_obs.ndim == 1:
        y_obs = np.tile(y_obs, (M, 1))
    
    # Background mean and anomalies
    mean_b = np.mean(X_b, axis=0)
    X_anom = X_b - mean_b  # (M, N)
    
    # Background covariance (N x N)
    P_b = (X_anom.T @ X_anom) / (M - 1)
    
    # Observation operator: H extracts observed indices
    # P_bH: (N, L), P_byy: (L, L)
    P_bH = P_b[:, obs_idx]  # (N, L)
    P_byy = P_b[np.ix_(obs_idx, obs_idx)]  # (L, L)
    
    # Innovation covariance
    S = P_byy + obs_var * np.eye(L)
    
    # Localized cross-covariance for better conditioning
    state_idx = np.arange(N)
    diffs = np.abs(state_idx[:, None] - obs_idx[None, :])
    distances = np.minimum(diffs, N - diffs)  # (N, L)
    
    L_loc = 2.0  # correlation length scale
    loc_matrix = np.exp(-distances / L_loc)  # (N, L)
    P_bH_loc = P_bH * loc_matrix  # (N, L)
    
    # Adaptive Tikhonov regularization
    # Estimate the optimal regularization from the ratio of observed to unobserved spread
    obs_spread = np.std(X_b[:, obs_idx], axis=0, ddof=1)
    unobs_idx = np.setdiff1d(np.arange(N), obs_idx)
    if len(unobs_idx) > 0:
        unobs_spread = np.std(X_b[:, unobs_idx], axis=0, ddof=1)
        # Use the ratio to adapt regularization strength
        spread_ratio = np.mean(obs_spread) / (np.mean(unobs_spread) + 1e-10)
        # Stronger regularization when unobserved spread is larger relative to observed
        _adaptive_alpha = np.clip(0.5 + 0.5 * spread_ratio, 0.3, 3.0)
    else:
        _adaptive_alpha = 1.0
    
    # EnKI: minimize J(u) = (1/L) * ||y_obs - H(u)||^2 / obs_var + (1/N) * ||u - u_b||^2 / (alpha * P_b_diag)
    # where u is the state vector, u_b is the background mean
    
    # We'll use an ensemble-based approach: each member is a candidate solution
    # and we perform gradient descent on the ensemble mean
    
    # Compute the gradient of the cost function w.r.t. the state
    # J(x) = (1/(2*obs_var)) * ||y - H(x)||^2 + (1/(2*alpha*P_diag)) * ||x - x_b||^2
    
    P_diag = np.diag(P_b)
    P_diag_safe = np.maximum(P_diag, 1e-10)
    
    # Number of gradient descent steps
    n_steps = 10
    
    # Learning rate schedule: start larger, decay
    base_lr = 0.15
    
    # Current state (ensemble mean)
    x = mean_b.copy()
    
    # Store the ensemble for reconstruction
    X_current = X_b.copy()
    
    for step in range(n_steps):
        # Compute H(x) - observed components
        x_obs = x[obs_idx]  # (L,)
        
        # Innovation: y - H(x)
        innovation = y_obs[0] - x_obs  # (L,)
        
        # Gradient of data misfit term
        # dJ_data/dx = -H^T * (y - H(x)) / obs_var
        # H is a selection operator, so H^T maps (L,) to (N,)
        grad_data = np.zeros(N)
        grad_data[obs_idx] = -innovation / obs_var
        
        # Gradient of regularization term
        # dJ_reg/dx = (x - x_b) / (alpha * P_diag)
        grad_reg = (x - mean_b) / (_adaptive_alpha * P_diag_safe)
        
        # Total gradient
        grad = grad_data + grad_reg
        
        # Adaptive learning rate based on gradient magnitude
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Normalize gradient for stable updates
        grad_normalized = grad / grad_norm
        
        # Learning rate with decay
        lr = base_lr * (0.8 ** step)
        
        # Update
        x = x + lr * grad_normalized * grad_norm  # equivalent to x + lr * grad
        
        # Clip to reasonable bounds
        x = np.clip(x, -100.0, 100.0)
    
    # Now reconstruct the ensemble around the updated mean
    # Preserve the ensemble spread structure but shift to new mean
    mean_a = x
    
    # Compute the analysis anomalies using the Kalman gain as a guide
    # for the correlation structure
    try:
        S_inv_PbH = np.linalg.solve(S.T, P_bH_loc.T)  # (L, N)
        K = S_inv_PbH.T  # (N, L)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K = P_bH_loc @ S_inv  # (N, L)
    
    # Member innovations
    X_obs_members = X_b[:, obs_idx]  # (M, L)
    member_innovations = y_obs - X_obs_members  # (M, L)
    
    # Kalman-filter-like update for anomalies
    X_anom_a = X_anom + member_innovations @ K.T  # (M, N)
    
    # Reconstruct ensemble: new mean + updated anomalies
    X_a = mean_a[None, :] + X_anom_a  # (M, N)
    
    # Adaptive multiplicative inflation to maintain spread
    spread_b = np.std(X_b, axis=0, ddof=1)
    spread_a = np.std(X_a, axis=0, ddof=1)
    
    min_ratio = 0.85
    inflation_factors = np.ones(N)
    for i in range(N):
        if spread_b[i] > 1e-6 and spread_a[i] > 1e-6:
            ratio = spread_a[i] / spread_b[i]
            if ratio < min_ratio:
                inflation_factors[i] = min(1.25, min_ratio / ratio)
    
    if np.any(inflation_factors > 1.0):
        X_a = mean_a[None, :] + inflation_factors[None, :] * (X_a - mean_a[None, :])
    
    # Safety: ensure finite and clip
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    np.clip(X_a, -100.0, 100.0, out=X_a)
    
    return X_a

# EVOLVE-BLOCK-END
