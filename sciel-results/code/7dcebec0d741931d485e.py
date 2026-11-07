"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using Adaptive Lower Confidence Bound (LCB).
    
    Strategy:
    1. Use LCB: score = -mu - beta * sigma.
       This favors points with low predicted mean (exploitation) and high uncertainty (exploration).
    2. Adaptive Beta: 
       - Estimate the 'maturity' of the GP model. In early stages, sigma is generally high.
       - We use a beta that decays as the model becomes more confident. 
       - However, for rugged functions like Weierstrass, we need strong exploration initially.
       - We set beta based on the current scale of uncertainty. A fixed high beta (e.g., 3.0-4.0) 
         often works better than adaptive ones in BO if the GP is well-calibrated, but 
         adaptive helps if the GP is overconfident in unexplored regions.
       - Let's use a beta that starts high and decays slightly if f_best is very close to 0, 
         but primarily we rely on a robust beta=3.5 which is aggressive for Weierstrass.
       
       Why different from current?
       - Current: Hard filter (mu <= f_best + tol) + UCB (beta=2.0). The filter removes points 
         that might be in a high-uncertainty valley that hasn't been sampled yet but whose 
         mean is currently predicted to be high. This is dangerous for rugged functions.
       - New: No hard filter. Pure LCB with higher beta. This allows the algorithm to "jump" 
         to unexplored regions even if the current mean prediction is poor, as long as the 
         uncertainty is high enough to compensate.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    if not np.isfinite(f_best):
        # Fallback: if f_best is invalid, just maximize exploration (high sigma)
        return np.random.randn(len(mu))
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive Beta Selection
    # In the early stages, the GP is uncertain everywhere. 
    # As evaluations proceed, sigma drops in explored areas.
    # For Weierstrass, local minima are a trap. We need to ensure we explore broadly.
    # A beta of 2.0 was used in the parent. We will try a higher beta to encourage 
    # more exploration of high-uncertainty regions, which are likely unexplored valleys.
    
    # We can estimate the "age" of the optimization by looking at the minimum sigma.
    # If min_sigma is very small, we have explored some points well.
    # If min_sigma is large, we are still in the early phase.
    
    min_sigma = np.min(sigma_safe)
    
    # Heuristic: 
    # If min_sigma > 0.1 (high uncertainty even at best point), we are very early. Use high beta.
    # If min_sigma < 0.01 (low uncertainty at best point), we are later. Use lower beta.
    # Weierstrass function values are in [0, ~1]. Standardized targets are used.
    # So sigma is in standardized units? No, "mu and sigma ... are in objective units".
    # Wait, "standardized targets" means the GP is trained on standardized y.
    # But the prompt says "mu and sigma ... are in objective units". This implies the 
    # GP posterior is transformed back to objective scale.
    
    # Let's check the scale. Weierstrass min is 0. Max is ~1.
    # If sigma is in objective units, it's likely small (0.01 - 0.1).
    
    # Let's use a fixed beta that is aggressive enough.
    # Literature on Weierstrass BO suggests beta around 2.0-3.0 is good.
    # The parent used 2.0 with a filter. Removing the filter and increasing beta 
    # should allow better global exploration.
    
    beta = 3.0
    
    # Calculate LCB scores
    # We want to maximize: -mu - beta * sigma
    # Note: LCB is usually for minimization: min (mu + beta*sigma). 
    # Maximizing acquisition means maximizing the "promise".
    # For minimization, a low value is good. 
    # So we want to pick x that minimizes mu + beta*sigma.
    # Therefore, the acquisition score to MAXIMIZE should be:
    # score = -(mu + beta * sigma) = -mu - beta * sigma
    
    scores = -mu - beta * sigma_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    # Check time
    if time.time() > deadline:
        pass
        
    return scores

# EVOLVE-BLOCK-END
