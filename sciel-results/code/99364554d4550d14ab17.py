import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Determine the current step index. 
    # The function doesn't receive 't' directly. 
    # However, the evaluator calls this 25 times after 5 initial points.
    # We can infer the step count from the number of points in the GP? 
    # No, 'mu' and 'sigma' are just arrays for 2000 candidates. 
    # We don't have access to the history length inside 'acquisition' unless we store it in global state.
    
    # Since we cannot pass 't' as an argument, we must maintain state.
    # We can use a global variable to count calls.
    
    if not hasattr(acquisition, 'call_count'):
        acquisition.call_count = 0
        
    t = acquisition.call_count
    acquisition.call_count += 1
    
    # Total BO steps expected is 25.
    # The first 5 points are initial random, not via BO.
    # So t goes from 0 to 24.
    
    T = 25
    
    # Decay exploration parameter xi
    # Start with high exploration, end with low.
    # xi=0 is pure exploitation.
    # Let's try a quadratic decay.
    progress = t / T
    xi = 0.5 * (1.0 - progress) ** 2
    
    # Calculate Expected Improvement with exploration offset xi
    # z = (f_best - xi - mu) / sigma
    # EI = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    
    # Handle numerical stability
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    
    z = (f_best - xi - mu) / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - xi - mu) * cdf + sigma_safe * pdf
    
    # Ensure finite scores
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
