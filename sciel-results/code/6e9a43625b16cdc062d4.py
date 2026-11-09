"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 300.0  # 300s limit per evaluation, leave margin

def _check_deadline():
    """Return True if we should stop and return current best."""
    return (time.time() - _START_TIME) > (_DEADLINE_SECONDS - 5.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Strategy:
    - Exploit: Prefer points with low predicted objective value (mu).
    - Explore: Prefer points with high GP uncertainty (sigma).
    - Cost-aware: Scale the entire score by 1/cost to maximize "improvement per unit cost".
    - Dynamic Beta: The exploration coefficient (beta) starts high to encourage broad search
      and decays as the budget is consumed (inferred from f_best improvement or simply fixed 
      if budget info is not directly available, but here we use a heuristic based on 
      the spread of mu and sigma to adapt).
      
    Since we don't have the exact iteration count or total budget remaining directly in the 
    signature, we estimate the "phase" of the optimization. 
    However, a robust UCB with a fixed or slowly varying beta is often superior to 
    complex adaptive schemes in short-budget scenarios. We will use a beta that 
    balances the scale of mu and sigma.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions in objective units
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best (lowest) observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition values (higher = better to evaluate)
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clamp sigma to avoid division by zero or extreme values
    sigma = np.maximum(sigma, 1e-8)
    
    # Check deadline
    if _check_deadline():
        # Return a simple, safe score to exit quickly
        # Higher score for lower mu (better) and higher sigma (more uncertain)
        return (-mu + sigma) / cost

    # Compute the UCB score.
    # We want to MINIMIZE f(x). 
    # Standard UCB for minimization: Score = -mu + beta * sigma
    # Why? We want high score for low mu (exploitation) and high sigma (exploration).
    # -mu is high when mu is low.
    # beta * sigma is high when sigma is high.
    
    # Determine beta (exploration coefficient)
    # A common heuristic is to scale beta relative to the range of the observed data 
    # or use a constant. Since mu and sigma are in objective units, and Shekel 4D 
    # has a range of roughly [-10.5, 10], a beta around 1.0-2.0 is typical.
    # To make it adaptive, we can look at the current f_best.
    # If f_best is very good (close to -10.5), we might want to exploit more (lower beta).
    # If f_best is poor, we might want to explore more (higher beta).
    
    # Let's use a simple adaptive beta based on the potential improvement.
    # The "potential" is roughly the distance from f_best to the known global min (-10.5364).
    # However, we don't know the global min exactly in the code, but we can estimate 
    # the "scale" of the problem.
    
    # A more robust approach: Normalize mu and sigma? No, they are in objective units.
    # Let's use a beta that decays slightly if f_best is very low, but stays reasonable.
    # For Shekel 4D, the objective values are in the range of -10 to 10.
    # Let's set beta = 1.5 as a strong exploration factor initially, which is often 
    # effective for multimodal functions in low dimensions with limited budgets.
    
    beta = 1.5
    
    # Calculate raw UCB score
    ucb_score = -mu + beta * sigma
    
    # Cost-awareness: Divide by cost.
    # This prioritizes points that offer high UCB value per unit cost.
    score = ucb_score / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Optional: Add a tiny bit of noise or randomization? 
    # The prompt says "score 2000 new uniform candidates... and evaluates the best one".
    # Deterministic selection is fine.
    
    return score
# EVOLVE-BLOCK-END
