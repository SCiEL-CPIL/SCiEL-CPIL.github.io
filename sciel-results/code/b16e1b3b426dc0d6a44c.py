"""Lorenz-96 assimilation using Ensemble Smoother with Multiple Scattering (ES-MSC)."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_DEADLINE = 145.0
_t0 = None

def _check_deadline():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def _lorenz96_rhs(x, F):
    """Compute the Lorenz-96 right-hand side for a single state vector."""
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

def _rk4_propagate(ens, F, dt, n_steps):
    """Propagate ensemble by n_steps RK4 steps. ens is (n_members, n_state)."""
    n_members, n_state = ens.shape
    for _ in range(n_steps):
        for m in range(n_members):
            ens[m] = _rk4_step(ens[m], F, dt)
    return ens

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Smoother with Multiple Scattering (ES-MSC) analysis update for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    ens = ensemble.astype(np.float64).copy()
    obs_idx = np.asarray(observed_indices)
    R = observation_variance
    
    # We need to estimate F. Since we don't have direct access, we'll use
    # a heuristic: the forcing F affects the mean level. We'll try to infer
    # it from the ensemble statistics, or use a default.
    # Actually, the problem states F is given by the evaluator's propagation.
    # We don't have F explicitly. Let's estimate it from the ensemble.
    # For Lorenz-96, the mean state scales with F. A rough estimate:
    # The variance of the state is roughly proportional to F^2.
    # But this is tricky. Let's assume we can't propagate without F.
    # 
    # Alternative approach: Since we can't propagate without knowing F,
    # we'll use a simpler non-linear update that doesn't require propagation.
    # We'll use a multiple scattering approach on the linear update itself,
    # iterating the EnSRF update multiple times with relaxation.
    
    # Precompute localization weights
    decay_len = 1.5
    loc_weights = np.zeros((n_obs, n_state))
    for j, oidx in enumerate(obs_idx):
        for i in range(n_state):
            d = abs(int(i) - int(oidx))
            d = min(d, n_state - d)
            loc_weights[j, i] = np.exp(-d / decay_len)
    
    # Multiple scattering: iterate the sequential EnSRF update
    # with a relaxation parameter to converge to a fixed point
    n_scatter = 3
    relaxation = 0.5
    
    for scatter in range(n_scatter):
        if _check_deadline():
            break
        
        # Save the pre-scatter state for relaxation
        ens_pre = ens.copy()
        
        # Sequential EnSRF: process observations one at a time
        for j in range(n_obs):
            oidx = int(obs_idx[j])
            obs_val = observations[j]
            
            # Current ensemble mean
            x_obs = ens[:, oidx]
            x_obs_mean = np.mean(x_obs)
            
            # Innovation
            innov = obs_val - x_obs_mean
            
            # Variance of observed variable in ensemble
            var_obs = np.var(x_obs, ddof=1)
            
            if var_obs < 1e-10:
                continue
            
            denom = var_obs + R
            
            # Update all coordinates
            for i in range(n_state):
                w = loc_weights[j, i] if i != oidx else 1.0
                if w < 0.01 and i != oidx:
                    continue
                
                x_i = ens[:, i]
                x_i_mean = np.mean(x_i)
                
                # Compute covariance
                if i == oidx:
                    cov_io = var_obs
                else:
                    cov_io = np.cov(x_i, x_obs, ddof=1)[0, 1]
                
                K = w * cov_io / denom
                
                new_mean = x_i_mean + K * innov
                new_anoms = (1 - K) * (x_i - x_i_mean)
                ens[:, i] = new_mean + new_anoms
        
        # Relaxation: blend between pre-scatter and post-scatter
        if scatter < n_scatter - 1:
            ens = (1 - relaxation) * ens_pre + relaxation * ens
    
    # Adaptive inflation
    ens_mean_new = np.mean(ens, axis=0)
    
    total_ratio = 0.0
    count = 0
    for j, oidx in enumerate(obs_idx):
        x_obs_new = ens[:, int(oidx)]
        var_new = np.var(x_obs_new, ddof=1)
        if var_new < 1e-10:
            continue
        ratio = R / var_new
        total_ratio += ratio
        count += 1
    
    if count > 0:
        mean_ratio = total_ratio / count
        inflation = np.sqrt(np.clip(mean_ratio, 0.9, 1.5))
        inflation = np.clip(inflation, 0.95, 1.3)
        
        ens = ens_mean_new + inflation * (ens - ens_mean_new)
    
    # Final safety
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
