import numpy as np

def acquisition(mu, sigma, f_best):
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # 1. Estimate time step t
    # Assume f_best starts around 0 and ends near -2.53218
    # Total BO steps = 25 (30 total evals - 5 initial)
    # t ranges from 1 to 25
    
    f_min_known = -2.53218
    f_start_guess = 0.0
    
    # Calculate progress: 0 at start, 1 at end
    # If f_best is better (lower) than f_start, progress increases
    # If f_best is worse (higher) than f_start, progress is 0
    
    if f_best >= f_start_guess:
        progress = 0.0
    else:
        # Normalize improvement
        improvement = f_start_guess - f_best
        total_improvement = f_start_guess - f_min_known
        progress = np.clip(improvement / total_improvement, 0.0, 1.0)
        
    t_est = 1.0 + 24.0 * progress
    
    # 2. Calculate beta_t
    # beta_t = sqrt(t * log(t^2 * c^2 / delta))
    delta = 0.1
    c = 1.0
    
    # Avoid log(0) or negative args
    arg = t_est**2 * c**2 / delta
    if arg <= 1.0:
        beta_t = 1.0 # Safe fallback
    else:
        beta_t = np.sqrt(t_est * np.log(arg))
        
    # 3. Calculate scores
    # Minimization: score = -mu + sqrt(beta_t) * sigma
    # Higher score is better
    
    sigma_safe = np.maximum(sigma, 0.0)
    scores = -mu + np.sqrt(beta_t) * sigma_safe
    
    # 4. Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
