"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer to enforce the 330s wall-clock limit with a safety margin
_start_time = time.time()
TIME_LIMIT = 320.0 # Leave 10s buffer for cleanup/return

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Uses a dynamic UCB strategy:
    Score = (mu + kappa * sigma) / cost
    kappa adapts based on the current uncertainty relative to the current best.
    """
    # Check time limit
    if time.time() - _start_time > TIME_LIMIT:
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive to avoid division by zero in z-score calculations
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive kappa for UCB.
    # In standard UCB, kappa is a constant. Here, we scale it based on how far 
    # the current mu is from f_best. If mu is already good (close to f_best), 
    # we explore less. If mu is bad, we might explore more to find a better region,
    # but primarily we want to balance exploitation (low mu) and exploration (high sigma).
    # A common heuristic for adaptive beta in BO is beta_t = 2 * log(t + 1) or similar.
    # Since we don't have 't' directly, we use the spread of sigma relative to f_best.
    # f_best is the minimum observed. mu should ideally be < f_best for improvement.
    
    # Calculate a scale for uncertainty. 
    # If f_best is 0 (perfect), sigma is noise. 
    # If f_best is large, sigma should be weighted more.
    # Let's use a simpler, robust adaptive kappa:
    # kappa = 2.0 is a common starting point for UCB in BO.
    # We can modulate it: if the model is very uncertain (high mean sigma), increase exploration.
    mean_sigma = np.mean(sigma_safe)
    
    # Normalize sigma to a typical scale to determine confidence level
    # If f_best is very low (close to 0), we are likely near the optimum.
    # We want to be conservative.
    
    # Strategy: 
    # 1. Compute base UCB: mu + kappa * sigma
    # 2. Divide by cost.
    
    # To handle the "minimization" aspect naturally:
    # We want HIGH scores for LOW (mu - kappa*sigma) ? No, UCB maximizes.
    # In minimization, we want to find points with low f(x).
    # The predictive distribution is Normal(mu, sigma^2).
    # We expect f(x) to be around mu.
    # To exploit, we want low mu.
    # To explore, we want high sigma.
    # Standard UCB for maximization: mu + beta*sigma.
    # For minimization, we can minimize mu - beta*sigma? 
    # Or maximize -(mu - beta*sigma) = beta*sigma - mu?
    # Let's stick to the form that favors low values.
    # Acquisition = -mu + beta * sigma ? 
    # If we maximize this, we prioritize low mu and high sigma.
    
    # However, the cost term is in the denominator.
    # Score = (-mu + beta * sigma) / cost ?
    # If mu is very negative (good), -mu is positive. Score is high.
    # If sigma is high, score is high.
    
    # Let's refine the beta (kappa).
    # A fixed beta of 2.0 to 3.0 usually works well for Levy functions.
    # Let's try an adaptive beta that decreases as we get closer to the optimum (f_best -> 0).
    # This encourages exploitation when we are close, and exploration when we are far.
    
    # Estimate "closeness" to optimum.
    # Since min is 0, if f_best is small, we are close.
    # Let's define a confidence parameter.
    
    # To be safe and distinct from the previous EI, we use UCB.
    # Score = (mu + kappa * sigma) / cost is for MAXIMIZATION.
    # For MINIMIZATION, we want to maximize the probability of finding a value lower than f_best?
    # No, simple UCB heuristic for minimization:
    # We want to pick x that minimizes f(x).
    # We estimate f(x) ~ N(mu, sigma^2).
    # A conservative estimate of the minimum possible value is mu - kappa*sigma.
    # We want to MAXIMIZE this "lower bound" to ensure we find a low value?
    # No, that's minimax regret.
    
    # Let's look at the standard transformation.
    # Maximization: Score = mu + beta*sigma.
    # Minimization: Score = -mu + beta*sigma.
    # Why? Because we want low mu. -mu is high when mu is low.
    # And we want high sigma to explore.
    
    # Let's use beta = 2.5 as a baseline for Levy 2D.
    # We will make beta adaptive: beta = 2.0 + 2.0 * (f_best / (f_best + 1.0))
    # If f_best is 0 (perfect), beta=2.0.
    # If f_best is large (bad), beta approaches 4.0.
    # This increases exploration when we are doing poorly.
    
    f_best_val = float(f_best)
    if f_best_val < 0:
        f_best_val = 0.0
        
    # Adaptive exploration constant
    # When f_best is high, we need to explore more to find the basin.
    # When f_best is low, we exploit more.
    kappa = 2.0 + 3.0 * (f_best_val / (f_best_val + 1.0))
    
    # Calculate UCB for minimization: -mu + kappa * sigma
    ucb_scores = -mu + kappa * sigma_safe
    
    # Normalize by cost
    # cost is positive.
    # We want to penalize high cost.
    scores = ucb_scores / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
