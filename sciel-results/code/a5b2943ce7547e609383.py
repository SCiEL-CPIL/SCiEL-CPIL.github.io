"""Lorenz-96 assimilation using Ensemble Smoother with Multiple Scattering (ES-MSC)."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Time deadline guard
_DEADLINE = 145.0
_t0 = None

def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE


def _lorenz96_step(x, F, dt=0.01, n_steps=5):
    """Propagate a single state vector through Lorenz-96 dynamics using RK4."""
    n = len(x)
    for _ in range(n_steps):
        k1 = _lorenz96_deriv(x, F, n)
        k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F, n)
        k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F, n)
        k4 = _lorenz96_deriv(x + dt * k3, F, n)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def _lorenz96_deriv(x, F, n):
    """Compute Lorenz-96 derivatives."""
    dx = np.empty_like(x)
    for i in range(n):
        dx[i] = (x[(i + 1) % n] - x[(i - 2) % n]) * x[(i - 1) % n] - x[i] + F
    return dx


def _lorenz96_step_ensemble(ens, F, dt=0.01, n_steps=5):
    """Propagate an ensemble of states through Lorenz-96 dynamics using RK4."""
    n_members, n_state = ens.shape
    for _ in range(n_steps):
        k1 = _lorenz96_deriv_batch(ens, F, n_state)
        k2 = _lorenz96_deriv_batch(ens + 0.5 * dt * k1, F, n_state)
        k3 = _lorenz96_deriv_batch(ens + 0.5 * dt * k2, F, n_state)
        k4 = _lorenz96_deriv_batch(ens + dt * k3, F, n_state)
        ens = ens + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return ens


def _lorenz96_deriv_batch(ens, F, n_state):
    """Compute Lorenz-96 derivatives for a batch of ensemble members."""
    n_members = ens.shape[0]
    dx = np.empty_like(ens)
    for i in range(n_state):
        dx[:, i] = (ens[:, (i + 1) % n_state] - ens[:, (i - 2) % n_state]) * ens[:, (i - 1) % n_state] - ens[:, i] + F
    return dx


def _es_msc_update(ens, observations, observed_indices, observation_variance, F, n_scatter=3):
    """
    Perform ES-MSC update: iteratively apply linear EnSRF update and re-propagate.
    
    Parameters:
    - ens: (n_members, n_state) ensemble
    - observations: (n_obs,) observed values
    - observed_indices: array of observed index positions
    - observation_variance: scalar
    - F: forcing parameter
    - n_scatter: number of scattering iterations
    
    Returns:
    - Updated ensemble
    """
    n_members, n_state = ens.shape
    n_obs = len(observations)
    
    # Estimate F from the ensemble if not provided
    if F is None:
        # Estimate F from mean dynamics: at steady state, mean(dx) ≈ 0
        # But we don't have F directly. Use a heuristic: F ≈ mean(ensemble) for L96
        # Actually, we need to infer F. Let's use the relation that for L96,
        # the mean value is approximately F. Use ensemble mean as proxy.
        F = float(np.mean(ens))
    
    # Number of scattering steps
    # Use a moderate number to balance accuracy and computation
    n_scatter = min(n_scatter, 3)  # Cap at 3 for safety
    
    for scatter in range(n_scatter):
        # Compute ensemble mean and covariance
        ens_mean = np.mean(ens, axis=0)
        anomalies = ens - ens_mean
        cov = (anomalies.T @ anomalies) / (n_members - 1)
        
        # Observation error covariance
        R = np.eye(n_obs) * observation_variance
        
        # Get observed state components from ensemble
        ens_obs = ens[:, observed_indices]  # (n_members, n_obs)
        ens_obs_mean = np.mean(ens_obs, axis=0)
        
        # Compute innovation
        innovation = observations - ens_obs_mean
        
        # Compute innovation covariance S = H P H^T + R
        P_obs_obs = cov[np.ix_(observed_indices, observed_indices)]
        S = P_obs_obs + R
        
        # Regularize S
        S_reg = S + 1e-8 * np.eye(n_obs)
        
        # Compute Kalman gain K = P H^T S^{-1}
        PHt = cov[:, observed_indices]  # (n_state, n_obs)
        
        try:
            K = np.linalg.solve(S_reg, PHt.T).T
        except np.linalg.LinAlgError:
            K = PHt @ np.linalg.pinv(S_reg)
        
        # Apply localization to the Kalman gain
        decay_length = 2.0
        localized_K = np.zeros_like(K)
        for j in range(n_obs):
            obs_idx = observed_indices[j]
            for i in range(n_state):
                d = abs(i - obs_idx)
                d = min(d, n_state - d)
                w = np.exp(-d ** 2 / (2.0 * decay_length ** 2))
                localized_K[i, j] = K[i, j] * w
        
        # Update each ensemble member
        member_innovations = observations[None, :] - ens_obs  # (n_members, n_obs)
        update = member_innovations @ localized_K.T  # (n_members, n_state)
        ens = ens + update
        
        # Adaptive inflation based on spread matching
        ens_mean_new = np.mean(ens, axis=0)
        anomalies_new = ens - ens_mean_new
        cov_new = (anomalies_new.T @ anomalies_new) / (n_members - 1)
        
        # Expected innovation variance after update
        P_obs_obs_new = cov_new[np.ix_(observed_indices, observed_indices)]
        S_expected = P_obs_obs_new + R
        
        # Compute inflation factor
        ratios = np.ones(n_obs)
        for j in range(n_obs):
            exp_var = S_expected[j, j]
            if exp_var > 1e-10:
                ratios[j] = (innovation[j] ** 2 + observation_variance) / exp_var
        
        log_ratios = np.log(np.clip(ratios, 1e-6, 1e6))
        geo_mean_ratio = np.exp(np.mean(log_ratios))
        inflation = np.sqrt(np.clip(geo_mean_ratio, 0.95, 1.25))
        
        # Apply inflation around the new mean
        ens = ens_mean_new + inflation * (ens - ens_mean_new)
        
        # Re-propagate through dynamics for next scattering iteration
        if scatter < n_scatter - 1:
            ens = _lorenz96_step_ensemble(ens, F, dt=0.01, n_steps=5)
    
    # Final clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an Ensemble Smoother with Multiple Scattering (ES-MSC) analysis update.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    n_members, n_state = ensemble.shape
    
    # Work with a copy
    ens = ensemble.astype(np.float64).copy()
    
    # Estimate forcing parameter F from the ensemble
    # For Lorenz-96, the mean value is approximately related to F
    # Use a robust estimate: F ≈ mean of the ensemble
    F = float(np.mean(ens))
    
    # Perform ES-MSC update
    ens = _es_msc_update(ens, observations, observed_indices, observation_variance, F, n_scatter=3)
    
    # Final safety checks
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
