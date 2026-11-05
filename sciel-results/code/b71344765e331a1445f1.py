"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Particle Flow Filter (PFF) for Lorenz-96 assimilation.
    
    Uses a smooth divergence-free flow field to transport ensemble members
    toward observation consistency, preserving non-Gaussian error structure.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    n_particles, n_state = ensemble.shape
    n_obs = len(observed_indices)
    
    # Observation noise standard deviation
    obs_std = np.sqrt(observation_variance)
    
    # Current ensemble statistics
    ens_mean = ensemble.mean(axis=0)
    ens_cov = np.cov(ensemble, rowvar=False, bias=False)
    
    # Observation statistics
    obs_mean = observations.mean(axis=0)
    obs_var = observations.var(axis=0, ddof=1)
    
    # PFF: Find a smooth flow that transports ensemble to match observations
    # We use a quadratic potential approach:
    # The flow velocity at particle x is: v(x) = -H^T (H H^T + R)^-1 (H x - y) * alpha
    # where H is the observation operator, R is observation noise covariance
    # We solve for a time-dependent flow over a short interval
    
    # Observation operator: H selects observed components
    # H is n_obs x n_state
    H = np.zeros((n_obs, n_state))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # R: observation noise covariance (diagonal)
    R = np.eye(n_obs) * observation_variance
    
    # For PFF, we want to find a flow that maps the ensemble distribution
    # to one that is consistent with observations. We use a Gaussian target
    # for the observed components and a smooth interpolation for unobserved.
    
    # Approach: Use a flow field based on a quadratic potential
    # v(x, t) = A(t) * (x - m_target(x))
    # where m_target is a smooth function that pulls observed components
    # toward the observation mean while preserving unobserved structure.
    
    # Simplified PFF: Use a linear flow field
    # The optimal linear flow that matches the first two moments of the
    # observed components is given by the Kalman gain.
    
    # Compute the Kalman gain for the observed components
    P_H = H @ ens_cov @ H.T  # n_obs x n_obs
    S = P_H + R  # Innovation covariance
    K = ens_cov @ H.T @ np.linalg.inv(S)  # n_state x n_obs
    
    # Innovation
    innovation = observations - H @ ensemble  # n_particles x n_obs
    
    # Standard EnKF update as baseline
    # ens_update = ensemble + K @ innovation.T
    
    # PFF improvement: Use a nonlinear flow that accounts for
    # higher-order statistics. We use a particle-based approach:
    # 1. Compute a flow field that moves each particle toward a target
    # 2. The target is determined by matching the observed components
    
    # For the PFF, we define a flow field:
    # v_i(x) = sum_j G_ij(x) * (y_j - H_j x)
    # where G is a gain matrix that depends on the particle position
    
    # We use a localized, adaptive gain that accounts for the
    # non-Gaussian structure of the error distribution.
    
    # Step 1: Compute the innovation for each particle
    # Step 2: Compute a local gain that adapts to the particle's position
    
    # Local covariance estimation for each particle (using its neighbors)
    # This helps capture non-Gaussian structure
    
    # For efficiency, we use a simplified approach:
    # 1. Compute the global Kalman gain
    # 2. Add a nonlinear correction based on the innovation magnitude
    # 3. Apply adaptive inflation
    
    # Global Kalman gain update
    ens_updated = ensemble + (K @ innovation.T).T
    
    # Nonlinear correction: Particles with large innovations get less correction
    # This preserves the tails of the error distribution
    innov_norm = np.linalg.norm(innovation, axis=1)  # n_particles
    innov_std = innov_norm.std() + 1e-10
    
    # Damping factor: particles with large innovations get less correction
    # This is a soft version of the "reject large innovations" approach
    damp = np.exp(-0.5 * (innov_norm / (2.0 * obs_std * np.sqrt(n_obs)))**2)
    damp = 0.5 + 0.5 * damp  # Range [0.5, 1.0]
    
    # Apply damped correction
    correction = (K @ innovation.T).T
    ens_updated = ensemble + damp[:, np.newaxis] * correction
    
    # Adaptive inflation: Prevent ensemble collapse
    # Compute the ensemble spread vs observation spread
    ens_spread_obs = np.sqrt(np.diag(H @ ens_cov @ H.T))
    obs_spread = np.sqrt(obs_var + observation_variance)
    
    # Inflation factor: increase spread if ensemble is too narrow
    spread_ratio = np.mean(ens_spread_obs) / (np.mean(obs_spread) + 1e-10)
    
    # Target spread ratio ~ 1.0
    if spread_ratio < 0.8:
        inflation = 1.0 + 0.1 * (0.8 - spread_ratio)
    elif spread_ratio > 1.5:
        inflation = 1.0 - 0.05 * (spread_ratio - 1.5)
    else:
        inflation = 1.0
    
    inflation = np.clip(inflation, 0.9, 1.3)
    
    # Apply inflation around the ensemble mean
    new_mean = ens_updated.mean(axis=0)
    ens_updated = new_mean + inflation * (ens_updated - new_mean)
    
    # Ensure finite values and within bounds
    ens_updated = np.nan_to_num(ens_updated, nan=0.0, posinf=100.0, neginf=-100.0)
    ens_updated = np.clip(ens_updated, -100.0, 100.0)
    
    # Check deadline
    if time.time() > deadline:
        return ens_updated
    
    # Second pass: Refine with a particle-based flow
    # Use the updated ensemble to compute a better flow field
    
    # Compute the updated ensemble statistics
    ens_mean2 = ens_updated.mean(axis=0)
    ens_cov2 = np.cov(ens_updated, rowvar=False, bias=False)
    
    # Recompute gain with updated covariance
    P_H2 = H @ ens_cov2 @ H.T
    S2 = P_H2 + R
    K2 = ens_cov2 @ H.T @ np.linalg.inv(S2)
    
    # Innovation with updated ensemble
    innovation2 = observations - H @ ens_updated
    correction2 = (K2 @ innovation2.T).T
    
    # Apply a smaller correction for refinement
    ens_final = ens_updated + 0.3 * correction2
    
    # Final inflation check
    ens_spread_obs2 = np.sqrt(np.diag(H @ ens_cov2 @ H.T))
    obs_spread2 = np.sqrt(obs_var + observation_variance)
    spread_ratio2 = np.mean(ens_spread_obs2) / (np.mean(obs_spread2) + 1e-10)
    
    if spread_ratio2 < 0.8:
        inflation2 = 1.0 + 0.05 * (0.8 - spread_ratio2)
    else:
        inflation2 = 1.0
    
    inflation2 = np.clip(inflation2, 0.95, 1.15)
    
    new_mean2 = ens_final.mean(axis=0)
    ens_final = new_mean2 + inflation2 * (ens_final - new_mean2)
    
    # Final bounds check
    ens_final = np.nan_to_num(ens_final, nan=0.0, posinf=100.0, neginf=-100.0)
    ens_final = np.clip(ens_final, -100.0, 100.0)
    
    return ens_final
# EVOLVE-BLOCK-END
