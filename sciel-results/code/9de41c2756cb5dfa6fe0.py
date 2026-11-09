"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute Lorenz-96 derivatives for a 20-variable state using vectorized ops."""
    n = len(x)
    # dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rk4_propagate_ensemble(ens, F, dt, steps):
    """Propagate each ensemble member forward by 'steps' RK4 steps of size dt."""
    for _ in range(steps):
        for i in range(ens.shape[0]):
            ens[i] = _rk4_step(ens[i], F, dt)
    return ens


def _infer_F(ensemble):
    """
    Infer the forcing F from the ensemble mean.
    At equilibrium for Lorenz-96, x_eq = F for all i.
    The ensemble mean approximates F.
    """
    return float(np.mean(ensemble))


def _build_localization_matrix(n, decay_scale=2.0):
    """
    Build a 20x20 localization matrix for Lorenz-96 ring geometry.
    Uses Gaussian decay based on ring distance.
    """
    L = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(i - j)
            d = min(d, n - d)  # ring distance
            L[i, j] = np.exp(-0.5 * (d / decay_scale) ** 2)
    return L


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter update with localization, inflation, and optional
    short-range model propagation for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of observed_indices
    - observed_indices: array of indices that were observed (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble: (16, 20) array
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit
    
    n_members, n_vars = ensemble.shape
    ens = ensemble.copy().astype(np.float64)
    
    # Determine observed and unobserved indices
    obs_idx = np.array(sorted(observed_indices))
    all_idx = np.arange(n_vars)
    unobs_idx = np.array([i for i in all_idx if i not in set(obs_idx.tolist())])
    
    # Infer F from ensemble mean
    F = _infer_F(ensemble)
    
    # --- Step 1: Short-range model propagation (1-2 RK4 steps) ---
    # The evaluator propagates 5 steps of dt=0.01 between updates.
    # We can propagate a fraction to account for drift.
    # Try 1 step first (conservative) to avoid over-propagating.
    dt = 0.01
    n_prop_steps = 1
    if time.time() - start_time < DEADLINE - 10:
        ens = _rk4_propagate_ensemble(ens, F, dt, n_prop_steps)
    
    # --- Step 2: Compute ensemble mean and covariance ---
    mean = ens.mean(axis=0)
    centered = ens - mean
    cov = (centered.T @ centered) / (n_members - 1)
    
    # --- Step 3: Apply localization to covariance ---
    # Build full localization matrix and apply element-wise
    decay_scale = 2.0
    L_full = _build_localization_matrix(n_vars, decay_scale)
    
    # Localized covariance: P_loc[i,j] = P[i,j] * L[i,j]
    cov_loc = cov * L_full
    
    # Extract blocks
    cov_oo = cov_loc[np.ix_(obs_idx, obs_idx)]
    cov_uo = cov_loc[np.ix_(unobs_idx, obs_idx)]  # unobs vs obs
    
    # --- Step 4: Kalman gain computation ---
    # Observation noise covariance R (diagonal)
    n_obs = len(obs_idx)
    R = np.eye(n_obs) * observation_variance
    
    # Innovation covariance: S = H P H^T + R
    S = cov_oo + R
    
    # Regularize S for numerical stability
    S += np.eye(n_obs) * 1e-10
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Innovation
    innovation = observations - mean[obs_idx]
    
    # Kalman gain for observed: K_oo = cov_oo @ S_inv
    K_oo = cov_oo @ S_inv
    # Kalman gain for unobserved: K_uo = cov_uo @ S_inv
    K_uo = cov_uo @ S_inv
    
    # Update vectors
    update_oo = K_oo @ innovation  # (n_obs,)
    update_uo = K_uo @ innovation  # (n_unobs,)
    
    # --- Step 5: Apply deterministic update to all ensemble members ---
    # Perturbed EnKF style: add the same update to each member
    for i in range(n_members):
        ens[i, obs_idx] += update_oo
        ens[i, unobs_idx] += update_uo
    
    # --- Step 6: Variance inflation to prevent filter collapse ---
    # Compute current spread
    new_mean = ens.mean(axis=0)
    current_spread = ens.std(axis=0)
    
    # Target spread: use pre-update covariance diagonal as reference
    pre_update_cov_diag = np.diag(cov)
    target_spread = np.sqrt(pre_update_cov_diag) * 1.03  # slight inflation
    
    for j in range(n_vars):
        if current_spread[j] > 1e-10:
            scale = target_spread[j] / current_spread[j]
            # Cap inflation to avoid instability
            scale = np.clip(scale, 0.8, 1.5)
            ens[:, j] = new_mean[j] + (ens[:, j] - new_mean[j]) * scale
        else:
            # If spread is zero, add small perturbation
            ens[:, j] = new_mean[j] + np.random.normal(0, 0.1, n_members)
    
    # --- Step 7: Safety checks ---
    if not np.all(np.isfinite(ens)):
        ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    ens = np.clip(ens, -100.0, 100.0)
    
    # Final deadline check
    if time.time() - start_time > DEADLINE:
        # Return whatever we have (already clipped and finite)
        pass
    
    return ens

# EVOLVE-BLOCK-END
