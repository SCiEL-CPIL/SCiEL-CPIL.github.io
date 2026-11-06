import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system using vectorized operations."""
    x_ip1 = np.roll(x, -1)
    x_im1 = np.roll(x, 1)
    x_im2 = np.roll(x, 2)
    return (x_ip1 - x_im2) * x_im1 - x + F

def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _compute_localization_matrix(D, observed_indices, L):
    """
    Compute Gaussian localization weights.
    D: total dimensions (20)
    observed_indices: indices of observed variables (10 indices)
    L: localization length scale
    Returns: (D, M) matrix of weights.
    """
    M = len(observed_indices)
    loc = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = abs(d - obs_i)
            dist = min(dist, D - dist) # Periodic distance
            loc[d, m_idx] = np.exp(-0.5 * (dist / L) ** 2)
    return loc

def _ensemble_analysis(ensemble, obs_vec, observed_indices, observation_variance,
                       localization_length=3.5):
    """
    Ensemble Kalman Filter analysis step.
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    obs_vec = np.asarray(obs_vec).flatten()
    
    # 1. Compute ensemble mean and anomalies
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (N, D)
    
    # 2. Observation-space anomalies
    obs_anoms = anomalies[:, observed_indices]  # (N, M)
    
    # 3. Covariances
    # P_oo: (M, M)
    P_oo = (obs_anoms.T @ obs_anoms) / (N - 1)
    # P_xo: (D, M)
    P_xo = (anomalies.T @ obs_anoms) / (N - 1)
    
    # 4. Localization
    # Apply localization to P_xo. This reduces spurious correlations.
    loc = _compute_localization_matrix(D, observed_indices, localization_length)
    P_xo_loc = P_xo * loc
    
    # 5. Innovation Covariance S = P_oo + R
    R = observation_variance * np.eye(M)
    S = P_oo + R
    
    # 6. Kalman Gain
    # K = P_xo_loc @ S^{-1}
    try:
        S_inv = np.linalg.inv(S)
        K = P_xo_loc @ S_inv
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K = P_xo_loc @ S_inv
        
    # 7. Innovations
    # y - Hx_ens
    ens_obs = ensemble[:, observed_indices]
    innovations = obs_vec[None, :] - ens_obs  # (N, M)
    
    # 8. Update
    # delta_x = innovations @ K.T
    delta_x = innovations @ K.T  # (N, D)
    updated_ensemble = ensemble + delta_x
    
    # 9. Adaptive Inflation
    # Check if the ensemble spread is appropriate.
    # Expected innovation variance: mean of diag(S)
    expected_innov_var = np.mean(np.diag(S))
    # Actual innovation variance: mean of squared innovations
    actual_innov_var = np.mean(innovations ** 2)
    
    if expected_innov_var > 1e-10:
        ratio = actual_innov_var / expected_innov_var
        # If ratio > 1, ensemble is under-dispersed (innovations too large)
        # If ratio < 1, ensemble is over-dispersed
        # We want to adjust spread so that ratio ~ 1.
        # Inflate/Deflate anomalies around the *analysis* mean.
        analysis_mean = np.mean(updated_ensemble, axis=0)
        
        # Target inflation factor
        # If ratio is 2, we want to double the spread -> factor sqrt(2)?
        # Variance scales with alpha^2. So alpha = sqrt(ratio).
        inflation = np.sqrt(ratio)
        
        # Clamp inflation to reasonable bounds to avoid instability
        inflation = np.clip(inflation, 0.8, 2.0)
        
        if inflation != 1.0:
            updated_anoms = updated_ensemble - analysis_mean
            updated_ensemble = analysis_mean + inflation * updated_anoms
    
    # Safety
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for partial-observation Lorenz-96.
    """
    start_time = time.time()
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0].copy()
    else:
        obs_vec = obs_arr.flatten().copy()
    
    # Ensure obs_vec matches M
    if len(obs_vec) != M:
        if len(obs_vec) > M:
            obs_vec = obs_vec[:M]
        else:
            padded = np.zeros(M)
            padded[:len(obs_vec)] = obs_vec
            obs_vec = padded
    
    # Run analysis
    result = _ensemble_analysis(
        ensemble, obs_vec, observed_indices, observation_variance,
        localization_length=3.5
    )
    
    # Final safety
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 145.0:
        return result
    
    return result

# EVOLVE-BLOCK-END
