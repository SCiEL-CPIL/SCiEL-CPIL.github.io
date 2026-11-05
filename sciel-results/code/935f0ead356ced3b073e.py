"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute RHS of Lorenz-96 system. x is (N, 20) or (20,)."""
    xm1 = np.roll(x, 1, axis=-1)
    xp1 = np.roll(x, -1, axis=-1)
    xm2 = np.roll(x, 2, axis=-1)
    return (xp1 - xm2) * xm1 - x + F


def _lorenz96_step(ensemble, F, dt=0.01, n_steps=5):
    """
    Propagate each ensemble member using RK4 for n_steps at dt.
    ensemble: (N, 20)
    F: forcing
    """
    x = ensemble.copy()
    N = x.shape[0]
    
    for _ in range(n_steps):
        k1 = _lorenz96_rhs(x, F)
        k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_rhs(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def _ensemble_mean(ensemble):
    return np.mean(ensemble, axis=0)


def _ensemble_cov(ensemble):
    """Compute ensemble covariance matrix (20, 20)."""
    N = ensemble.shape[0]
    mean = _ensemble_mean(ensemble)
    centered = ensemble - mean
    cov = centered.T @ centered / (N - 1)
    return cov


def _localization(cov, decay=1.5, dim=20):
    """
    Apply periodic exponential localization to the covariance matrix.
    decay: decay rate (larger = more localization)
    """
    idx = np.arange(dim)
    dist = np.minimum(np.abs(idx[:, None] - idx[None, :]), dim - np.abs(idx[:, None] - idx[None, :]))
    weights = np.exp(-decay * dist)
    return cov * weights


def _inflation(ensemble, inflation_factor=1.01):
    """
    Apply variance inflation to keep ensemble spread healthy.
    Returns inflated ensemble.
    """
    mean = _ensemble_mean(ensemble)
    spread = np.std(ensemble, axis=0)
    # Inflate around the mean
    inflated = mean + inflation_factor * (ensemble - mean)
    return inflated


def _ensvar_update(ensemble, observations, observed_indices, obs_var, 
                    localization_decay=1.5, inflation=1.005, 
                    max_iterations=10, tol=1e-6):
    """
    Ensemble Variance (EnVar) analysis using iterative minimization.
    This is a variational assimilation approach that minimizes a cost function
    combining background and observation misfits.
    """
    N, dim = ensemble.shape
    mean = _ensemble_mean(ensemble)
    
    # Background covariance with localization
    B = _ensemble_cov(ensemble)
    B_loc = _localization(B, decay=localization_decay, dim=dim)
    
    # Add small regularization for numerical stability
    B_reg = B_loc + 1e-6 * np.eye(dim)
    
    # Invert background covariance (Cholesky for stability)
    try:
        L = np.linalg.cholesky(B_reg)
        B_inv = np.linalg.inv(B_reg)
    except np.linalg.LinAlgError:
        B_inv = np.linalg.pinv(B_reg)
    
    # Observation operator: selects observed indices
    H = np.zeros((len(observed_indices), dim))
    H[np.arange(len(observed_indices)), observed_indices] = 1.0
    
    # Observation covariance
    R = obs_var * np.eye(len(observed_indices))
    
    # Initial guess: background mean
    x = mean.copy()
    
    # Precompute terms
    H_B_inv = H @ B_inv  # (nobs, dim)
    R_inv = np.linalg.inv(R)
    HR_inv = R_inv @ H  # (nobs, dim)
    
    # H R^-1 H^T B^-1
    H_R_inv_Ht_B_inv = H @ R_inv @ H.T @ B_inv  # (dim, dim)
    
    # Gradient of cost function:
    # J(x) = 0.5 (x-xb)^T B^-1 (x-xb) + 0.5 (y - Hx)^T R^-1 (y - Hx)
    # dJ/dx = B^-1 (x - xb) - H^T R^-1 (y - Hx)
    
    prev_cost = np.inf
    for it in range(max_iterations):
        # Compute gradient
        resid = observations - H @ x
        grad = B_inv @ (x - mean) - H.T @ (R_inv @ resid)
        
        # Compute Hessian approximation (use B^-1 + H^T R^-1 H as it's positive definite)
        # For efficiency, use fixed-point iteration / Gauss-Seidel style
        # x_new = xb + B H^T R^-1 (y - H xb) - B H^T R^-1 H (x - xb)
        # Simplified: use iterative scheme
        
        # Gauss-Newton style update
        # x_{k+1} = x_k - alpha * grad
        # Use line search with backtracking
        alpha = 1.0
        for ls in range(20):
            x_trial = x - alpha * grad
            # Compute cost at trial
            diff_b = x_trial - mean
            diff_o = observations - H @ x_trial
            cost = 0.5 * np.sum(diff_b @ B_inv @ diff_b) + 0.5 * np.sum(diff_o @ R_inv @ diff_o)
            if cost < prev_cost or np.isnan(prev_cost):
                break
            alpha *= 0.5
        else:
            alpha = 0.01
        
        x_new = x - alpha * grad
        x_new = np.clip(x_new, -100, 100)
        
        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    
    # Update ensemble: shift members to new mean, preserve spread structure
    # But also adjust spread based on analysis covariance
    # Simple approach: shift the ensemble by the analysis increment
    increment = x - mean
    updated_ensemble = ensemble + increment[None, :]
    
    # Apply light inflation to prevent collapse
    updated_ensemble = _inflation(updated_ensemble, inflation_factor=inflation)
    
    return updated_ensemble


def _ensemble_kalman(ensemble, observations, observed_indices, obs_var,
                      localization_decay=1.5, inflation=1.005):
    """
    Ensemble Kalman Filter (EnKF) update.
    """
    N, dim = ensemble.shape
    mean = _ensemble_mean(ensemble)
    
    # Background covariance with localization
    B = _ensemble_cov(ensemble)
    B_loc = _localization(B, decay=localization_decay, dim=dim)
    
    # Observation operator
    H = np.zeros((len(observed_indices), dim))
    H[np.arange(len(observed_indices)), observed_indices] = 1.0
    
    # Innovation covariance: S = H B H^T + R
    S = H @ B_loc @ H.T + obs_var * np.eye(len(observed_indices))
    
    # Innovation
    y = observations
    y_pred = H @ mean
    innovation = y - y_pred
    
    # Kalman gain: K = B H^T S^-1
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = B_loc @ H.T @ S_inv  # (dim, nobs)
    
    # Analysis mean
    mean_a = mean + K @ innovation
    
    # Update ensemble members
    # Use the ensemble-based approach: perturb observations and propagate
    # Or use the deterministic update with spread preservation
    
    # Perturbation-based EnKF
    # Generate perturbed observations
    rng = np.random.default_rng(42)
    updated_ensemble = np.zeros_like(ensemble)
    for i in range(N):
        # Perturb observation for this member
        if i == 0:
            y_pert = y
        else:
            y_pert = y + rng.normal(0, np.sqrt(obs_var), size=len(observations))
        
        innovation_i = y_pert - y_pred
        x_i = ensemble[i] + K @ innovation_i
        updated_ensemble[i] = x_i
    
    # Apply inflation
    updated_ensemble = _inflation(updated_ensemble, inflation_factor=inflation)
    
    return updated_ensemble


def _hybrid_update(ensemble, observations, observed_indices, obs_var,
                    localization_decay=1.2, inflation=1.003):
    """
    Hybrid approach: combine EnVar and EnKF.
    Use EnVar for the mean update, then adjust ensemble spread.
    """
    # First, do an EnVar-like update for the mean
    N, dim = ensemble.shape
    mean = _ensemble_mean(ensemble)
    
    # Background covariance with localization
    B = _ensemble_cov(ensemble)
    B_loc = _localization(B, decay=localization_decay, dim=dim)
    B_reg = B_loc + 1e-5 * np.eye(dim)
    
    try:
        B_inv = np.linalg.inv(B_reg)
    except np.linalg.LinAlgError:
        B_inv = np.linalg.pinv(B_reg)
    
    # Observation operator
    H = np.zeros((len(observed_indices), dim))
    H[np.arange(len(observed_indices)), observed_indices] = 1.0
    
    # Observation covariance
    R = obs_var * np.eye(len(observed_indices))
    R_inv = np.linalg.inv(R)
    
    # Innovation
    y = observations
    y_pred = H @ mean
    innovation = y - y_pred
    
    # Analysis increment using variational formulation
    # delta_x = B H^T (H B H^T + R)^-1 innovation
    S = H @ B_loc @ H.T + R
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    delta_x = B_loc @ H.T @ (S_inv @ innovation)
    
    # New mean
    mean_a = mean + delta_x
    mean_a = np.clip(mean_a, -100, 100)
    
    # Update ensemble: shift by increment, adjust spread
    # Preserve relative structure but scale to match analysis spread
    # Analysis covariance: P_a = (I - K H) B (I - K H)^T + K R K^T
    K = B_loc @ H.T @ S_inv
    I_KH = np.eye(dim) - K @ H
    P_a = I_KH @ B_loc @ I_KH.T + K @ R @ K.T
    
    # Scale ensemble to match analysis covariance in observed directions
    # Simple approach: shift and apply mild inflation
    updated_ensemble = ensemble + delta_x[None, :]
    
    # Apply inflation to maintain spread
    updated_ensemble = _inflation(updated_ensemble, inflation_factor=inflation)
    
    # Clip to valid range
    updated_ensemble = np.clip(updated_ensemble, -100, 100)
    
    return updated_ensemble


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update using hybrid variational/Kalman approach.
    
    Parameters:
    -----------
    ensemble : ndarray of shape (16, 20)
        Ensemble of state vectors
    observations : ndarray of shape (10,)
        Noisy observations of even-indexed coordinates
    observed_indices : array-like of shape (10,)
        Indices of observed coordinates
    observation_variance : float
        Variance of observation noise
    
    Returns:
    --------
    updated_ensemble : ndarray of shape (16, 20)
        Updated ensemble after assimilation
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Ensure inputs are numpy arrays
    ensemble = np.array(ensemble, dtype=float)
    observations = np.array(observations, dtype=float)
    observed_indices = np.array(observed_indices, dtype=int)
    
    N, dim = ensemble.shape
    
    # Try the hybrid update
    try:
        updated = _hybrid_update(ensemble, observations, observed_indices, 
                                 observation_variance,
                                 localization_decay=1.0, inflation=1.002)
    except Exception:
        # Fallback to simple nudging if something goes wrong
        updated = ensemble.copy()
        updated[:, observed_indices] += 0.5 * (observations - updated[:, observed_indices])
    
    # Ensure finite values
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    updated = np.clip(updated, -100, 100)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Return immediately if we're close to timeout
        pass
    
    return updated

# EVOLVE-BLOCK-END
