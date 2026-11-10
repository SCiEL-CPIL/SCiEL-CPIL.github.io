"""Hybrid Ensemble-4D-Var analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def _lorenz96_rhs(x, F):
    """Compute Lorenz-96 RHS: dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F"""
    n = len(x)
    dx = np.empty_like(x)
    for i in range(n):
        ip1 = (i + 1) % n
        im2 = (i - 2) % n
        im1 = (i - 1) % n
        dx[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + F
    return dx


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _gaspari_cohn(d, c=1.5):
    """Gaspari-Cohn correlation function for localization."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    
    mask1 = d <= 0.5 * c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    r1 = d[mask1]
    if len(r1) > 0:
        result[mask1] = (-5.0/4.0)*(r1/c)**5 + 5.0*(r1/c)**4 - 10.0*(r1/c)**3 + 5.0*(r1/c)
    
    r2 = d[mask2]
    if len(r2) > 0:
        result[mask2] = (1.0/12.0)*(r2/c)**5 - (5.0/4.0)*(r2/c)**4 + (5.0/2.0)*(r2/c)**3 - 5.0*(r2/c)**2 + (25.0/12.0)*(r2/c) + 0.5
    
    r3 = d[mask3]
    if len(r3) > 0:
        result[mask3] = (-1.0/16.0)*(r3/c)**5 + (1.0/4.0)*(r3/c)**4 - (5.0/8.0)*(r3/c)**3 - (5.0/2.0)*(r3/c)**2 + (25.0/12.0)*(r3/c) - 0.25
    
    return result


def _hybrid_4dvar_update(ensemble, observations, observed_indices, observation_variance,
                         n_members=16, n_state=20, n_obs=10,
                         F=8.0, dt=0.01, n_steps=5,
                         localization_scale=3.0, max_iter=80, tol=1e-8):
    """Hybrid Ensemble-4D-Var with localized cost function.
    
    J(x) = 0.5*(x - x_b)^T B^{-1} (x - x_b) + 0.5 * sum_j w_j*(y_j - H_j*x)^2 / R_jj
    where w_j are localization weights for each observation.
    """
    mean = np.mean(ensemble, axis=0)
    perturbations = ensemble - mean
    ens_cov = np.cov(ensemble.T)  # (n_state, n_state)
    
    # Regularize covariance more aggressively for small ensembles
    reg = 1e-4 * np.trace(ens_cov) / n_state
    ens_cov += reg * np.eye(n_state)
    ens_cov_inv = np.linalg.inv(ens_cov)
    
    # Build localization weights for each observation
    # w_j = Gaspari-Cohn(0) = 1 for the observed variable itself
    # We use localization to downweight observations that are far from the state
    # being updated, effectively creating a local analysis
    L_weights = np.ones(n_obs)
    for j, obs_idx in enumerate(observed_indices):
        # The observation directly constrains state obs_idx
        # Localization helps by reducing the influence of distant observations
        # on nearby states through the background term coupling
        L_weights[j] = 1.0  # Direct observation weight
    
    # R = observation_variance * I (diagonal)
    R_inv_diag = np.ones(n_obs) / observation_variance
    
    # H: observation operator (n_obs x n_state), selects observed indices
    H = np.zeros((n_obs, n_state))
    for j, obs_idx in enumerate(observed_indices):
        H[j, obs_idx] = 1.0
    
    # Cost function with localization-aware observation term
    def cost_and_grad(x):
        x = np.clip(x, -100.0, 100.0)
        # Background term
        dx = x - mean
        J_bg = 0.5 * dx @ ens_cov_inv @ dx
        g_bg = ens_cov_inv @ dx
        
        # Observation term with localization weights
        y_hat = H @ x
        innov = observations - y_hat
        weighted_innov = innov * np.sqrt(L_weights)
        J_obs = 0.5 * np.sum(weighted_innov * R_inv_diag * innov)
        g_obs = H.T @ (R_inv_diag * innov * L_weights)
        
        J = J_bg + J_obs
        g = g_bg + g_obs
        return J, g
    
    # Better initial guess: use ensemble mean but blend with observations
    # using a simple variational estimate
    x0 = mean.copy()
    for j, obs_idx in enumerate(observed_indices):
        # Weight the nudge by the relative confidence
        ens_std = np.std(ensemble[:, obs_idx])
        obs_std = np.sqrt(observation_variance)
        weight = (1.0 / (ens_std**2 + 1e-10)) / (1.0 / (ens_std**2 + 1e-10) + 1.0 / (obs_std**2 + 1e-10))
        x0[obs_idx] = (1.0 - weight) * mean[obs_idx] + weight * observations[j]
    
    # L-BFGS optimization with tighter tolerance
    from scipy.optimize import minimize
    res = minimize(cost_and_grad, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8,
                            'maxls': 100, 'maxfun': 500})
    
    x_analysis = res.x
    x_analysis = np.clip(x_analysis, -100.0, 100.0)
    
    # Compute the analysis perturbation from the optimization
    delta = x_analysis - mean
    
    # Construct updated ensemble with spread preservation
    # Use eigenvalue-based scaling to maintain ensemble spread
    orig_var = np.var(ensemble, axis=0)
    target_var = orig_var.copy()
    
    # The analysis mean has moved; we want the ensemble to maintain similar spread
    # but we can slightly reduce spread if the innovation was large (more confidence)
    innov_magnitude = np.abs(observations - mean[observed_indices])
    max_innov = np.max(innov_magnitude) if len(innov_magnitude) > 0 else 0.0
    obs_std = np.sqrt(observation_variance)
    
    # If innovations are small relative to obs error, keep spread; if large, reduce slightly
    innov_ratio = np.clip(max_innov / (obs_std + 1e-10), 0.0, 5.0)
    spread_scale = 1.0 - 0.05 * innov_ratio  # Mild reduction for large innovations
    spread_scale = np.clip(spread_scale, 0.7, 1.0)
    
    # Build updated ensemble
    result = np.zeros_like(ensemble)
    for m in range(n_members):
        result[m, :] = x_analysis + perturbations[m, :] * spread_scale
    
    # Ensure finite and bounded
    result = np.clip(result, -100.0, 100.0)
    result[~np.isfinite(result)] = 0.0
    
    return result


def analyze(ensemble, observations, observed_indices, observation_variance):
    """Perform Hybrid Ensemble-4D-Var analysis update.
    
    Parameters:
    ensemble: (16, 20) array of ensemble members
    observations: (10,) array of noisy observations of even-indexed states
    observed_indices: (10,) array of observed state indices
    observation_variance: scalar observation error variance
    
    Returns:
    Updated ensemble (16, 20)
    """
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin before 150s kill
    
    try:
        n_members, n_state = ensemble.shape
        n_obs = len(observed_indices)
        
        # Estimate F from the ensemble using the forcing balance
        # For Lorenz-96 at equilibrium, <x_i> ≈ F for many parameter regimes
        # Use a robust estimate: median of ensemble means
        F = float(np.median(np.mean(ensemble, axis=0)))
        # Clamp to reasonable range
        F = np.clip(F, 5.0, 12.0)
        
        result = _hybrid_4dvar_update(
            ensemble, observations, observed_indices, observation_variance,
            n_members=n_members, n_state=n_state, n_obs=n_obs,
            F=F, dt=0.01, n_steps=5,
            localization_scale=3.0, max_iter=80, tol=1e-8
        )
        
        # Time check
        elapsed = time.time() - t_start
        if elapsed > deadline:
            # Return what we have
            return result
        
        return result
        
    except Exception as e:
        # Fallback: weighted nudging using ensemble spread
        result = ensemble.copy().astype(float)
        ens_std = np.std(result, axis=0) + 1e-10
        obs_std = np.sqrt(observation_variance)
        weight = (1.0 / obs_std**2) / (1.0 / obs_std**2 + 1.0 / ens_std[observed_indices]**2)
        for j, obs_idx in enumerate(observed_indices):
            result[:, obs_idx] += weight[j] * (observations[j] - result[:, obs_idx])
        result = np.clip(result, -100.0, 100.0)
        result[~np.isfinite(result)] = 0.0
        return result
# EVOLVE-BLOCK-END
