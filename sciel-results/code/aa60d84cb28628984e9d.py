"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_deriv(x, F):
    """Compute Lorenz-96 derivatives for a 20-variable state."""
    n = len(x)
    dx = np.zeros(n)
    for i in range(n):
        dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
    return dx

def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_deriv(x, F)
    k2 = _lorenz96_deriv(x + 0.5*dt*k1, F)
    k3 = _lorenz96_deriv(x + 0.5*dt*k2, F)
    k4 = _lorenz96_deriv(x + dt*k3, F)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def _rk4_propagate(x, F, dt, steps):
    """Propagate state x forward by 'steps' RK4 steps of size dt."""
    for _ in range(steps):
        x = _rk4_step(x, F, dt)
    return x

def _ensemble_mean(ens):
    return ens.mean(axis=0)

def _ensemble_cov(ens):
    """Compute covariance matrix of ensemble (members x variables)."""
    n_members, n_vars = ens.shape
    mean = ens.mean(axis=0)
    centered = ens - mean
    cov = (centered.T @ centered) / (n_members - 1)
    return cov

def _localize_cov(cov, obs_idx, unobs_idx, decay_scale=3.0):
    """Apply exponential localization to covariance blocks."""
    n = len(obs_idx)
    m = len(unobs_idx)
    # Build distance matrix for observed-observed, observed-unobserved, unobserved-unobserved
    # In Lorenz-96, indices are on a ring of 20, so distance is min(|i-j|, 20-|i-j|)
    
    def exp_local(i, j):
        d = abs(i - j)
        d = min(d, 20 - d)
        return np.exp(-0.5 * (d / decay_scale)**2)
    
    # Localize observed-observed block
    cov_oo = cov[np.ix_(obs_idx, obs_idx)].copy()
    for ii in range(len(obs_idx)):
        for jj in range(len(obs_idx)):
            cov_oo[ii, jj] *= exp_local(obs_idx[ii], obs_idx[jj])
    
    # Localize observed-unobserved block
    cov_ou = cov[np.ix_(obs_idx, unobs_idx)].copy()
    for ii in range(len(obs_idx)):
        for jj in range(len(unobs_idx)):
            cov_ou[ii, jj] *= exp_local(obs_idx[ii], unobs_idx[jj])
    
    # Localize unobserved-unobserved block
    cov_uu = cov[np.ix_(unobs_idx, unobs_idx)].copy()
    for ii in range(len(unobs_idx)):
        for jj in range(len(unobs_idx)):
            cov_uu[ii, jj] *= exp_local(unobs_idx[ii], unobs_idx[jj])
    
    return cov_oo, cov_ou, cov_uu

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter update with localization and inflation.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of observed_indices
    - observed_indices: array of indices that were observed (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble: (16, 20) array
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit
    
    n_members, n_vars = ensemble.shape
    ens = ensemble.copy()
    
    # Determine which indices are observed and unobserved
    obs_idx = np.array(sorted(observed_indices))
    all_idx = np.arange(n_vars)
    unobs_idx = np.array([i for i in all_idx if i not in set(obs_idx.tolist())])
    
    # Compute ensemble mean and covariance
    mean = ens.mean(axis=0)
    cov = _ensemble_cov(ens)
    
    # Localize covariance
    cov_oo, cov_ou, cov_uu = _localize_cov(cov, obs_idx, unobs_idx, decay_scale=3.0)
    
    # Observation noise covariance (diagonal)
    R = np.diag(np.full(len(obs_idx), observation_variance))
    
    # Innovation covariance: S = H P H^T + R
    S = cov_oo + R
    
    # Kalman gain: K = P H^T S^{-1}
    # For observed-observed: K_oo = cov_oo * S^{-1}
    # For unobserved-observed: K_uo = cov_ou^T * S^{-1}
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse
        S_inv = np.linalg.pinv(S)
    
    # Innovation
    innovation = observations - mean[obs_idx]
    
    # Update observed components
    K_oo = cov_oo @ S_inv  # shape (n_obs, n_obs)
    update_oo = K_oo @ innovation  # shape (n_obs,)
    
    # Update unobserved components
    K_uo = cov_ou.T @ S_inv  # shape (n_unobs, n_obs)
    update_uo = K_uo @ innovation  # shape (n_unobs,)
    
    # Apply updates to ensemble mean
    new_mean = mean.copy()
    new_mean[obs_idx] += update_oo
    new_mean[unobs_idx] += update_uo
    
    # Apply updates to ensemble members (perturbed Kalman filter style)
    # For each member, add the same deterministic update
    for i in range(n_members):
        ens[i, obs_idx] += update_oo
        ens[i, unobs_idx] += update_uo
    
    # Simple variance inflation to prevent filter collapse
    # Inflate the spread around the new mean
    current_spread = ens.std(axis=0)
    target_spread = np.sqrt(np.diag(cov_uu)) * 1.05  # slight inflation for unobserved
    # For observed, keep spread reasonable
    for j in range(n_vars):
        if j in unobs_idx:
            idx_in_unobs = list(unobs_idx).index(j)
            if target_spread[idx_in_unobs] > 0:
                scale = target_spread[idx_in_unobs] / (current_spread[j] + 1e-10)
                scale = min(scale, 2.0)  # cap inflation
                ens[:, j] = new_mean[j] + (ens[:, j] - new_mean[j]) * scale
        else:
            # Mild inflation for observed
            scale = 1.02
            ens[:, j] = new_mean[j] + (ens[:, j] - new_mean[j]) * scale
    
    # Check for non-finite values
    if not np.all(np.isfinite(ens)):
        ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    return ens

# EVOLVE-BLOCK-END
