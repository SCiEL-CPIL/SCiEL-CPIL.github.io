"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Kriging Believer (Probability of Improvement).
    
    This strategy ranks candidates by the probability that they will improve upon the current best,
    adjusted for the cost of evaluation. Unlike EI, this avoids the PDF calculation and relies
    on the CDF, which is often more stable for ranking purposes in high dimensions when combined
    with cost normalization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions.
    sigma : array of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x)).
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    DEADLINE = 325.0  # Safety margin before 330s limit
    
    # 1. Input Sanitization
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Floor sigma and cost to prevent division by zero or log issues
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1.0) # Cost is at least 1 by definition (1 + mean(x))

    # 2. Deadline Check (Early exit to simple heuristic if needed)
    if time.time() - t0 > DEADLINE - 5:
        # Fallback: Simple UCB
        scores = (f_best - mu + 1.0 * sigma) / cost
        return np.where(np.isfinite(scores), scores, 0.0)

    # 3. Kriging Believer Calculation
    # We want to maximize P(f(x) < f_best)
    # f(x) ~ N(mu, sigma^2)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Clip z to prevent numerical issues in CDF (though Phi is stable, 
    # extreme values can lead to 0 or 1 which might be less informative for ranking 
    # if we multiply by other terms, but here it's the primary driver)
    # However, for ranking, 0 and 1 are fine.
    # We use a small epsilon to avoid exact 0/1 if we were doing logs, but for direct score, it's fine.
    prob_improve = norm.cdf(z)
    
    # 4. Cost-Aware Normalization
    # The goal is to find the best value per unit cost.
    # Using 1/cost is the standard cost-aware acquisition (Cost-EI).
    # However, the "inspiration" and cell findings suggest 1/sqrt(cost) might be better
    # for exploring high-cost regions that contain the minimum.
    # Let's try a blend or a specific power. 
    # The current program uses cost**0.5 in denominator (i.e., 1/sqrt(cost)).
    # Let's stick to the finding: 1/sqrt(cost) reduced penalty on expensive evals.
    
    cost_penalty = np.sqrt(cost)
    
    # Score = Probability of Improvement / Cost Penalty
    scores = prob_improve / cost_penalty
    
    # 5. Refinement: Add a small uncertainty bonus to break ties or encourage exploration
    # If prob_improve is very similar, we might want to slightly favor higher sigma
    # to map the function better, but Kriging Believer is inherently exploitative.
    # To ensure we don't get stuck, we can add a small term.
    # However, pure KB is often sufficient. Let's add a tiny UCB component weighted by (1 - prob_improve)
    # to ensure we explore where we are most uncertain if the probability is low?
    # Actually, KB naturally handles this: high sigma -> z is less extreme -> prob is closer to 0.5.
    # Low sigma + mu < f_best -> z is large -> prob is close to 1.
    # Low sigma + mu > f_best -> z is negative -> prob is close to 0.
    
    # Let's ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final safety check
    if time.time() - t0 > DEADLINE - 2:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
