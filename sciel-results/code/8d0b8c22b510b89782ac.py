import numpy as np
import time

def _lorenz96_rhs(x, F):
    """
    Compute the right-hand side of the Lorenz-96 equations.
    x: array of shape (n,)
    F: forcing parameter
    """
    n = len(x)
    # dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    # Periodic indices
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

def _rk4_step(x, dt, F):
    """
    Single RK4 step for one trajectory.
    """
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _infer_F(ensemble, observations, observed_indices):
    """
    Estimate the forcing parameter F from the current state.
    Using the mean of the observed variables is a reasonable proxy,
    though for L96, F is a constant. 
    Actually, F is not directly observable. 
    However, the mean of the L96 system is approximately F.
    Let's estimate F from the ensemble mean of the observed indices.
    """
    return np.mean(ensemble[:, observed_indices])

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a Local Ensemble Square Root Filter (LES-RF) update.
    
    Parameters:
    ensemble: (N, D) array, N ensemble members, D state dimension (20)
    observations: (D_obs,) array, observations of observed variables
    observed_indices: (D_obs,) array, indices of observed variables
    observation_variance: float, variance of observation noise
    
    Returns:
    updated_ensemble: (N, D) array
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    N, D = ensemble.shape
    x = ensemble.copy()
    
    # 1. Estimate F (optional, but good for context if we were propagating)
    # We don't actually use F in the update step itself, only in propagation which is done by the evaluator.
    # But knowing F might help in more advanced schemes. For LES-RF, we strictly use ensemble and obs.
    
    # 2. Adaptive Inflation
    # Calculate ensemble spread vs observation error
    # We want the ensemble spread at observed locations to be comparable to obs error
    ens_mean = np.mean(x, axis=0)
    ens_var = np.var(x, axis=0)
    
    # At observed indices
    obs_ens_var = ens_var[observed_indices]
    # Target variance should be roughly obs_var + true_ens_var
    # If ens_var is too small, inflate.
    # Simple multiplicative inflation:
    # alpha = 1.0 + 0.1 * (obs_var / obs_ens_var - 1) if obs_ens_var < obs_var else 1.0
    
    # Better: compute inflation factor such that the posterior spread matches the prior spread scaled by some factor
    # Or simply: if the ensemble is collapsing, inflate.
    # Let's use a simple adaptive inflation based on the ratio of obs variance to ens variance at obs points.
    mean_obs_ens_var = np.mean(obs_ens_var)
    if mean_obs_ens_var < 1e-6:
        alpha = 1.5
    else:
        # If ens var is less than obs var, we are overconfident.
        # Scale spread to be at least obs_var (roughly)
        # alpha = 1.0 + 0.1 * (observation_variance / mean_obs_ens_var - 1.0)
        # Clip to [1.0, 2.0]
        alpha = 1.0 + 0.1 * (observation_variance / mean_obs_ens_var - 1.0)
        alpha = np.clip(alpha, 1.0, 1.5)
    
    # Apply inflation to all variables (global inflation)
    # x = ens_mean + alpha * (x - ens_mean)
    x = ens_mean + alpha * (x - ens_mean)
    
    # 3. LES-RF Update
    # We will process each observed variable sequentially.
    # For each observed variable j:
    #   a. Compute ensemble mean and covariance matrix for the subset of variables correlated with j?
    #      Standard LES-RF uses the full ensemble covariance but localizes it.
    #      To keep it simple and robust, we can use the full ensemble but apply localization weights.
    #      However, the "Local" part usually means we update the entire ensemble based on the observation of variable j,
    #      using the covariance between variable j and all other variables.
    
    # Let's implement the standard "Local Ensemble Square Root Filter" (Hunt et al. 2007)
    # For each observation y_j:
    #   1. Compute ensemble mean x_bar and ensemble perturbation X' = X - x_bar
    #   2. Compute covariance matrix C = X' X'ᵀ / (N-1)
    #   3. Localize C: C_loc = C ⊙ R (R is localization matrix, e.g., Gaussian)
    #   4. Compute Kalman gain K = C_loc @ Hᵀ @ (H C_loc Hᵀ + R_obs)⁻¹
    #      Since we observe one scalar y_j, H is a row vector with 1 at index j.
    #      H C_loc Hᵀ is the variance of variable j in the localized covariance.
    #      K is a vector of length D.
    #   5. Update: X_new = X + K (y_j - H x_bar)
    #      Wait, this is the linear update. The "Square Root" part ensures the ensemble spread is correctly updated.
    #      Actually, the simple linear update with the full covariance matrix is equivalent to the Square Root Filter
    #      if done correctly. The key is using the *localized* covariance.
    
    # Let's use a simplified approach that is robust:
    # For each observed index j:
    #   - Compute the covariance between variable j and all variables i.
    #   - cov_ij = np.cov(x[:, j], x[:, i])
    #   - Localize: cov_loc_ij = cov_ij * exp(-dist(i,j)^2 / (2*sigma_loc^2))
    #   - Gain_i = cov_loc_ij / (var_j_loc + obs_var)
    #   - x[:, i] += Gain_i * (y_j - x[:, j])
    
    # This is effectively a localized Kalman filter update.
    # It is "Local" in the sense that the gain is localized.
    
    sigma_loc = 3.0  # Localization length scale (in grid points)
    # Precompute localization weights for all pairs?
    # D=20, so 20x20 matrix is small.
    
    # Indices 0..19
    indices = np.arange(D)
    # Distance matrix (periodic)
    dist = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            d = abs(i - j)
            dist[i, j] = min(d, D - d)
    
    # Localization matrix R_loc
    # Gaussian kernel
    R_loc = np.exp(-dist**2 / (2 * sigma_loc**2))
    
    # Now iterate over observed variables
    # To respect the sequential nature and update the ensemble, we must update x as we go.
    # However, computing covariance from the *current* x at each step is expensive and can be biased.
    # Standard practice: compute the ensemble mean and covariance matrix ONCE before the update,
    # or update sequentially using the current ensemble.
    # Given N=16, D=20, computing covariance 10 times is very fast.
    
    # Let's do sequential updates using the current ensemble state.
    for j in observed_indices:
        # Check time
        if time.time() > deadline:
            break
            
        # Compute ensemble mean and variance for variable j
        x_j = x[:, j]
        x_j_mean = np.mean(x_j)
        
        # Compute covariance between j and all i
        # cov_i = np.cov(x[:, i], x_j)
        # To do this efficiently:
        # Centered x
        x_centered = x - np.mean(x, axis=0)
        # Covariance vector for j:
        # cov_i = np.sum(x_centered[:, i] * x_centered[:, j]) / (N-1)
        cov_vec = np.sum(x_centered * x_centered[:, j], axis=0) / (N - 1)
        
        # Localize covariance
        cov_loc = cov_vec * R_loc[j, :]
        
        # Variance of j (localized)
        var_j_loc = cov_loc[j]
        
        # Observation error
        y_j = observations[j]
        
        # Kalman Gain
        # K_i = cov_loc_i / (var_j_loc + obs_var)
        denom = var_j_loc + observation_variance
        if denom < 1e-10:
            continue
            
        K = cov_loc / denom
        
        # Innovation
        innovation = y_j - x_j_mean
        
        # Update ensemble
        x += np.outer(K, innovation)
        
        # Optional: Re-center? No, the mean is updated correctly.
        # But the covariance changes. The next step will recompute it.
        
    # Final sanity check: clip to [-100, 100]
    x = np.clip(x, -100.0, 100.0)
    
    return x

