# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter analysis update for Lorenz-96.
    Uses random rotation of ensemble anomalies to preserve spread and reduce spurious correlations.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices, dtype=np.int64)
    R_val = float(observation_variance)
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # --- Compute ensemble mean and anomalies ---
    ens_mean = np.mean(ens, axis=0)
    ens_anom = ens - ens_mean  # (n_members, n_vars)
    
    # --- Random Rotation of Anomalies ---
    # This is a structural change: instead of inflation, we rotate the ensemble
    # in the member space to break spurious correlations and maintain spread.
    # Generate a random Gaussian matrix and orthogonalize it.
    rng = np.random.default_rng()
    Z = rng.standard_normal((n_members, n_members))
    Q, _ = np.linalg.qr(Z)
    # Ensure Q is a proper rotation (det=1), though for EnKF this is less critical
    # but good for consistency.
    if np.linalg.det(Q) < 0:
        Q[0, :] *= -1
        
    # Rotate anomalies
    ens_anom_rot = ens_anom @ Q
    
    # Reconstruct rotated ensemble
    ens_rot = ens_mean + ens_anom_rot
    
    # --- Localization weights ---
    sigma = 3.0  # Slightly wider localization
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)  # (n_vars, n_obs)
    
    # --- Stochastic Ensemble Kalman Filter Update ---
    # Now we perform the standard EnKF update on the rotated ensemble.
    
    # Anomalies at observed locations (of the rotated ensemble)
    A_obs = ens_rot[:, obs_idx] - ens_mean[obs_idx]  # (n_members, n_obs)
    # Note: ens_mean is the same for rotated ensemble.
    
    # Innovation
    innovation = obs - ens_mean[obs_idx]  # (n_obs,)
    
    # Cross-covariance between all variables and observations
    P_xo = (ens_anom_rot.T @ A_obs) / (n_members - 1.0)  # (n_vars, n_obs)
    
    # Apply localization
    P_xo_loc = P_xo * loc_weights
    
    # Observation error variance (diagonal, independent)
    P_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # (n_obs, n_obs)
    S = P_oo + R_val * np.eye(n_obs)
    
    # Cholesky of S
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        jitter = 1e-8
        S_reg = S + jitter * np.eye(n_obs)
        L = np.linalg.cholesky(S_reg)
    
    # Compute gain in square-root form
    L_inv = np.linalg.inv(L)
    S_inv = L_inv.T @ L_inv  # (n_obs, n_obs)
    
    # Gain matrix
    K = P_xo_loc @ S_inv  # (n_vars, n_obs)
    
    # Update each member: x_a_i = x_f_i + K @ (obs - x_f_i[obs_idx])
    # Here x_f_i is the rotated forecast member
    member_innovations = obs[None, :] - ens_rot[:, obs_idx]  # (n_members, n_obs)
    
    # Update
    ens_updated = ens_rot + (K @ member_innovations.T).T  # (n_members, n_vars)
    
    # --- Post-processing: ensure finite and bounded ---
    ens_updated = np.clip(ens_updated, -100.0, 100.0)
    ens_updated = np.where(np.isfinite(ens_updated), ens_updated, 0.0)
    
    if time.time() > deadline:
        return ens_updated.astype(ensemble.dtype)
    
    return ens_updated.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
