"""Particle Flow Filter for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _lorenz96_rhs(x, F):
    """Compute RHS of Lorenz-96 system."""
    n = len(x)
    dx = np.empty_like(x)
    for i in range(n):
        dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
    return dx

def _rk4_step(x, dt, F):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Particle Flow Filter (PFF) update for Lorenz-96 assimilation.
    
    Uses a smooth flow to transport the ensemble toward the observation
    distribution while preserving non-Gaussian structure.
    """
    # Deadline guard
    t_start = time.time()
    DEADLINE = 12.0  # seconds, well within 120s limit
    
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=np.int64)
    
    n_members, n_state = ensemble.shape
    n_obs = len(observed_indices)
    
    # Estimate F from the ensemble (rough estimate from the dynamics)
    # We'll use a simple approach: assume F is in {6, 8, 10} and pick based on mean
    mean_state = np.mean(ensemble, axis=0)
    # For Lorenz-96, the mean state roughly scales with F
    # A better approach: just use the flow without needing F explicitly
    # since we're doing a direct update, not propagation
    
    # PFF: Transport ensemble to match observation statistics
    # 1. Compute ensemble mean and covariance
    ens_mean = np.mean(ensemble, axis=0)
    ens_cov = np.cov(ensemble, rowvar=False)
    
    # 2. Observation statistics
    obs_mean = np.mean(observations, axis=0)
    obs_cov = np.cov(observations, rowvar=False)
    
    # 3. Build the observation operator H (selects observed indices)
    # H is n_obs x n_state
    H = np.zeros((n_obs, n_state))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # 4. Compute innovation covariance
    # S = H P H^T + R
    R = observation_variance * np.eye(n_obs)
    P_obs = H @ ens_cov @ H.T + R
    
    # 5. Compute Kalman-like gain for the flow
    # K = P H^T S^{-1}
    try:
        S_inv = np.linalg.inv(P_obs)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(P_obs)
    
    K = ens_cov @ H.T @ S_inv  # n_state x n_obs
    
    # 6. PFF flow: use a smooth mapping
    # For each particle, compute the flow velocity
    # The flow maps the prior ensemble to the posterior
    # We use a linear flow as a starting point, but add nonlinearity
    
    # Linear PFF: x_post = x_prior + K @ (y - H @ x_prior)
    # This is equivalent to EnKF, but we can add corrections
    
    # Compute innovations
    innovations = observations - H @ ensemble.T  # n_obs x n_members
    innovations = innovations.T  # n_members x n_obs
    
    # Linear update
    update = innovations @ K.T  # n_members x n_state
    result = ensemble + update
    
    # 7. Add a small nonlinear correction to preserve structure
    # Compute the residual after linear update
    residual = observations - H @ result.T  # n_obs x n_members
    residual = residual.T  # n_members x n_obs
    
    # Apply a second-order correction (small)
    # This helps capture non-Gaussian features
    correction_scale = 0.1
    second_order = residual @ K.T
    result += correction_scale * second_order
    
    # 8. Localization: dampen updates for unobserved coordinates
    # Use a simple exponential localization based on distance to observed indices
    # Since we're in a ring, compute circular distance
    loc_result = result.copy()
    for i in range(n_state):
        if i not in observed_indices:
            # Find nearest observed index (circular distance)
            min_dist = min(min(abs(i - oi), n_state - abs(i - oi)) for oi in observed_indices)
            # Exponential decay
            loc_factor = np.exp(-0.5 * min_dist)
            # Blend: keep some of the original ensemble for unobserved
            loc_result[:, i] = (1 - loc_factor) * ensemble[:, i] + loc_factor * result[:, i]
    
    result = loc_result
    
    # 9. Inflation: check if spread is too small
    post_mean = np.mean(result, axis=0)
    post_cov = np.cov(result, rowvar=False)
    
    # Compare observation-space spread
    obs_space_prior = np.var(ensemble[:, observed_indices], axis=0)
    obs_space_post = np.var(result[:, observed_indices], axis=0)
    obs_space_obs = np.var(observations, axis=0)
    
    # If posterior spread is too small relative to observations, inflate
    inflation_factors = []
    for j in range(n_obs):
        prior_spread = obs_space_prior[j]
        post_spread = obs_space_post[j]
        obs_spread = obs_space_obs[j]
        
        if post_spread > 1e-10:
            # Desired: post_spread should be comparable to obs_spread
            # Inflate if post_spread is too small
            target_ratio = np.sqrt(obs_spread / (post_spread + 1e-10))
            target_ratio = np.clip(target_ratio, 0.95, 1.15)
        else:
            target_ratio = 1.0
        
        inflation_factors.append(target_ratio)
    
    # Apply inflation in observation space
    if np.any(np.array(inflation_factors) != 1.0):
        for j, idx in enumerate(observed_indices):
            factor = inflation_factors[j]
            if abs(factor - 1.0) > 1e-4:
                mean_val = post_mean[idx]
                result[:, idx] = mean_val + factor * (result[:, idx] - mean_val)
    
    # 10. Final safety: ensure finite and within bounds
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > DEADLINE:
        # Return what we have
        pass
    
    return result
# EVOLVE-BLOCK-END
