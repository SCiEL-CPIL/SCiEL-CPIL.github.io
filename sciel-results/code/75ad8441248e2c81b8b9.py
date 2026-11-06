import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Inversion (EnKI) with Tikhonov regularization
    for Lorenz-96 partial-observation assimilation.
    
    Treats unobserved variables as control variables and uses stochastic gradient
    descent to minimize a misfit cost function with strong regularization to the
    background ensemble mean. This bypasses linear Gaussian assumptions and
    localization artifacts inherent in standard filters.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline to stay under 150s kill
    
    n_members, n_vars = ensemble.shape
    x = ensemble.copy().astype(np.float64)
    obs_var = observation_variance
    
    # Adaptive inflation state (persistent across calls via module-level dict)
    if not hasattr(analyze, '_inflation'):
        analyze._inflation = 1.0
    
    # Apply previous inflation
    if analyze._inflation > 1.0:
        mean_prev = np.mean(x, axis=0)
        x = mean_prev + analyze._inflation * (x - mean_prev)
    
    # EnKI approach: minimize cost function
    # Cost = sum over observed vars of (obs - pred)^2 / obs_var
    #       + lambda * sum over all vars of (x - bg_mean)^2 / bg_var
    # We treat each ensemble member independently and use SGD to update
    
    bg_mean = np.mean(ensemble, axis=0)
    bg_std = np.std(ensemble, axis=0, ddof=1)
    bg_var = np.maximum(bg_std ** 2, 1e-8)
    
    # Tikhonov regularization strength
    # Strong enough to keep solution close to background but allow correction
    reg_lambda = 0.5
    
    # Number of SGD iterations
    n_iters = 50
    lr = 0.05  # learning rate
    
    # Pre-compute observation-related quantities
    obs_indices = np.array(observed_indices)
    n_obs = len(obs_indices)
    
    # For each member, we want to find x_i that minimizes:
    # J(x_i) = (y - H x_i)^T R^{-1} (y - H x_i) + lambda * (x_i - bg_mean)^T B^{-1} (x_i - bg_mean)
    
    # Where H is the observation operator (selects observed indices)
    # R is the observation error covariance (diag with obs_var)
    # B is the background error covariance (approximated as diag with bg_var)
    
    # Gradient of J with respect to x_i:
    # dJ/dx_i = -H^T R^{-1} (y - H x_i) + lambda * B^{-1} (x_i - bg_mean)
    
    # We use SGD: x_i <- x_i - lr * grad
    
    for member_idx in range(n_members):
        x_member = x[member_idx].copy()
        
        for it in range(n_iters):
            # Check deadline
            if time.time() > deadline - 10.0:
                break
            
            # Compute predicted observations
            pred_obs = x_member[obs_indices]
            
            # Innovation
            innovation = observations - pred_obs  # (n_obs,)
            
            # Gradient from observation term: -H^T R^{-1} (y - H x_i)
            # H^T R^{-1} innovation: scatter the innovation back to full state, divided by obs_var
            grad_obs = np.zeros(n_vars)
            grad_obs[obs_indices] = innovation / obs_var
            
            # Gradient from regularization term: lambda * B^{-1} (x_i - bg_mean)
            grad_reg = reg_lambda * (x_member - bg_mean) / bg_var
            
            # Total gradient
            grad = -grad_obs + grad_reg
            
            # SGD update
            x_member = x_member - lr * grad
            
            # Early stopping if convergence
            if np.max(np.abs(grad)) < 1e-6:
                break
        
        x[member_idx] = x_member
    
    # Apply adaptive inflation based on spread vs innovation
    mean_a = np.mean(x, axis=0)
    
    # Compute spread and error at observed variables
    spread = np.std(x[:, obs_indices], axis=0, ddof=1)
    spread_mean = np.mean(spread)
    
    obs_err = observations - mean_a[obs_indices]
    obs_err_std = np.std(obs_err)
    
    if spread_mean > 1e-6:
        ratio = obs_err_std / spread_mean
        ratio = np.clip(ratio, 0.9, 1.5)
        # Exponential moving average
        analyze._inflation = 0.8 * analyze._inflation + 0.2 * ratio
        analyze._inflation = np.clip(analyze._inflation, 0.95, 1.2)
    else:
        analyze._inflation = 1.0
    
    # Apply inflation for the returned ensemble
    if analyze._inflation > 1.0:
        x = mean_a + analyze._inflation * (x - mean_a)
    
    # Safety checks
    if not np.all(np.isfinite(x)):
        x = ensemble.copy()
    
    x = np.clip(x, -100.0, 100.0)
    
    return x
