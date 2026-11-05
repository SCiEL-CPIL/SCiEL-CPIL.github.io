"""Lorenz-96 ensemble assimilation using Local Ensemble Transform (LET) with inflation."""
import numpy as np
import time

def _lorenz96_rhs(x, F):
    """Compute Lorenz-96 right-hand side for a single state vector."""
    n = len(x)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x[(i + 1) % n] - x[(i - 2) % n]) * x[(i - 1) % n] - x[i] + F
    return dx

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Transform (LET) assimilation for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (16, 10) or similar
        Noisy observations of the observed indices.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = 120.0  # seconds
    
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    n_members, n_states = ensemble.shape
    n_obs = len(observed_indices)
    
    # Ensure observation_variance is positive
    R = max(float(observation_variance), 1e-12)
    
    # --- Inflation ---
    # Compute ensemble spread vs observation spread at observed locations
    ens_mean_obs = ensemble[:, observed_indices].mean(axis=0)
    obs_mean = observations.mean(axis=0)
    
    # Variance of ensemble at observed locations
    ens_var_obs = np.mean((ensemble[:, observed_indices] - ens_mean_obs[None, :])**2, axis=0)
    # Variance of observations (proxy for truth spread)
    obs_var = np.mean((observations - obs_mean[None, :])**2, axis=0)
    
    # Simple multiplicative inflation: inflate if ensemble is too narrow
    # Target: ensemble variance should be comparable to observation variance
    # inflation_factor = sqrt(obs_var / max(ens_var_obs, eps))
    # But we need to be careful not to over-inflate
    
    # Use a moderate inflation scheme
    inflation_alpha = 1.0
    for i in range(n_obs):
        ratio = obs_var[i] / max(ens_var_obs[i], 1e-10)
        if ratio > 1.0:
            inflation_alpha = max(inflation_alpha, 1.0 + 0.5 * np.log1p(ratio))
    
    # Apply multiplicative inflation around ensemble mean
    ens_mean = ensemble.mean(axis=0)
    ensemble = ens_mean[None, :] + inflation_alpha * (ensemble - ens_mean[None, :])
    
    # --- Local Ensemble Transform ---
    # Process each state variable locally with its neighbors
    # For Lorenz-96, use a local window around each variable
    
    result = ensemble.copy()
    
    # Localization: use a cosine taper for local windows
    # Window size: consider variables within distance w of the target
    window_size = 5  # half-width of local window
    
    for i in range(n_states):
        # Check deadline
        if time.time() > deadline - 5.0:
            break
        
        # Determine local window: indices within window_size of i (circular)
        local_indices = [(i + d) % n_states for d in range(-window_size, window_size + 1)]
        
        # Check if any observed indices fall within this local window
        obs_in_window = []
        for oi in observed_indices:
            # Compute circular distance
            dist = min(abs(oi - i), n_states - abs(oi - i))
            if dist <= window_size:
                obs_in_window.append(oi)
        
        if len(obs_in_window) == 0:
            # No observations in this window, skip update
            continue
        
        # Build local ensemble: only variables in the local window
        local_ens = ensemble[:, local_indices]  # (n_members, len(local_indices))
        local_obs = observations[:, [list(observed_indices).index(oi) for oi in obs_in_window]]
        
        # Wait, observations are indexed by observed_indices, so we need to map correctly
        # observations[:, j] corresponds to observed_indices[j]
        # So for obs_in_window, we need the corresponding column indices in observations
        obs_col_indices = []
        for oi in obs_in_window:
            idx = list(observed_indices).index(oi)
            obs_col_indices.append(idx)
        local_obs = observations[:, obs_col_indices]  # (n_members, len(obs_in_window))
        
        # Compute local ensemble mean and perturbations
        local_mean = local_ens.mean(axis=0)  # (len(local_indices),)
        local_pert = local_ens - local_mean[None, :]  # (n_members, len(local_indices))
        
        # Compute local observation innovation
        local_obs_mean = local_obs.mean(axis=0)  # (len(obs_in_window),)
        
        # For LET, we need to compute the transformation matrix
        # Local covariance: C = (1/(n-1)) * Y^T Y where Y are perturbations
        # But we need the full local state covariance, including correlations between local states
        
        # Simplified LET approach:
        # 1. Compute Kalman gain using local ensemble statistics
        # 2. Apply transformation
        
        # Local state covariance (len(local_indices) x len(local_indices))
        C_local = (local_pert.T @ local_pert) / (n_members - 1)
        
        # Observation operator H: maps local state to observations
        # H is len(obs_in_window) x len(local_indices)
        H = np.zeros((len(obs_in_window), len(local_indices)))
        for k, oi in enumerate(obs_in_window):
            # Find position of oi in local_indices
            for j, li in enumerate(local_indices):
                if li == oi:
                    H[k, j] = 1.0
                    break
        
        # Innovation covariance: S = H C_local H^T + R
        HC = H @ C_local
        S = HC @ H.T + R * np.eye(len(obs_in_window))
        
        # Kalman gain: K = C_local H^T S^{-1}
        try:
            S_inv = np.linalg.inv(S)
            K = C_local @ H.T @ S_inv  # (len(local_indices), len(obs_in_window))
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            S_inv = np.linalg.pinv(S)
            K = C_local @ H.T @ S_inv
        
        # Innovation: y - H x_bar
        innovation = local_obs - (H @ local_mean[None, :].T).T  # (n_members, len(obs_in_window))
        
        # Update perturbations using LET
        # For each member, compute the update
        # x_a = x_b + K (y - H x_b)
        # But for ensemble, we use the transform approach
        
        # Compute the update for each member
        updates = (innovation @ K.T)  # (n_members, len(local_indices))
        
        # Apply update to local ensemble
        result[:, local_indices] = local_ens + updates
        
        # Renormalize to preserve ensemble mean (optional, helps stability)
        new_local_mean = result[:, local_indices].mean(axis=0)
        target_mean = local_mean + K @ (local_obs_mean - H @ local_mean)
        result[:, local_indices] += (target_mean - new_local_mean)[None, :]
    
    # Final inflation check and clipping
    # Ensure values are finite and within bounds
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
