"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB)
    divided by cost.
    
    Args:
        mu: Mean predictions, shape (N,)
        sigma: Standard deviation predictions, shape (N,)
        f_best: Best observed value (scalar)
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        Scores for each candidate, shape (N,)
    """
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle potential zero or negative costs (though prompt says cost >= 1)
    # Use a small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # LCB: Lower Confidence Bound for minimization
    # We want to minimize, so we subtract uncertainty.
    # Beta is a hyperparameter controlling exploration vs exploitation.
    # 2.0 is a standard choice for balancing these in BO.
    beta = 2.0
    
    lcb = mu - beta * sigma
    
    # Normalize by cost to favor cheaper evaluations
    scores = lcb / cost_safe
    
    # Ensure finite scores
    # If any calculation results in NaN or Inf (e.g., due to extreme values),
    # replace with a very large negative number (since we maximize the score to pick next).
    # Wait, the task says "higher = evaluate next".
    # So we want high scores.
    # LCB is low when good (mu low, sigma low).
    # Dividing by cost: if cost is high, score is low (bad).
    # So we maximize (LCB / Cost).
    
    # However, LCB can be very negative.
    # If we maximize, we pick the highest value.
    # High LCB means: High mu (bad) OR High sigma (uncertain).
    # This is correct for UCB (maximize uncertainty/reward).
    # For minimization, LCB = mu - beta*sigma.
    # We want to minimize the function.
    # Standard UCB for maximization: mu + beta*sigma.
    # Standard LCB for minimization: mu - beta*sigma.
    # We select the point with the MINIMUM LCB?
    # NO. The prompt says: "Return 2000 finite scores (higher = evaluate next)".
    # So the acquisition function must return a value where HIGHER is BETTER.
    
    # If we use LCB = mu - beta*sigma:
    # A good point has LOW mu.
    # So LCB is LOW.
    # If we return LCB, the evaluator will pick the MAXIMUM LCB, which is the WORST point (high mu).
    # We must INVERT the LCB or use a different formulation.
    
    # Correct approach for "Higher is Better" score for minimization:
    # Use Negative LCB? Or use Probability of Improvement?
    # Or use Expected Improvement (EI) which is positive when improvement is expected.
    
    # Let's re-evaluate EI.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # EI is >= 0.
    # High EI means high potential improvement.
    # This fits "Higher is Better".
    
    # So, if I switch to LCB, I must return -LCB / cost?
    # -LCB = -mu + beta*sigma.
    # High -LCB means: High beta*sigma (uncertainty) OR Low -mu (high mu? No, -mu is high if mu is low/negative).
    # If mu is low (good), -mu is high (positive).
    # So -LCB is high for good points (low mu) and high uncertainty.
    # This works.
    
    # Let's use Negative LCB per unit cost.
    # Score = (-mu + beta * sigma) / cost
    
    scores = (-mu + beta * sigma) / cost_safe
    
    # Replace non-finite values
    # If sigma is huge, scores are huge. This is fine.
    # If cost is 0, handled by cost_safe.
    
    return scores
# EVOLVE-BLOCK-END
