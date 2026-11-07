# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition based on expected information gain per unit cost.
    
    The core idea is to maximize the mutual information between a candidate's
    objective value and the posterior over the global minimum, normalized by
    evaluation cost. A practical surrogate for this information gain is the
    Expected Improvement (EI), which captures how much we expect to improve
    upon f_best. We enhance this with an uncertainty-weighted term that
    rewards high posterior variance in regions where improvement is plausible,
    and then divide by cost to penalize expensive evaluations.
    
    The score is:
      score = (EI + alpha * sigma * exp(-beta * max(mu - f_best, 0))) / cost
    
    where:
      - EI is the standard Expected Improvement
      - alpha * sigma * exp(-beta * max(mu - f_best, 0)) is an exploration bonus
        that is large when sigma is high and mu is near or below f_best
      - alpha controls the strength of exploration
      - beta controls how quickly the exploration bonus decays as mu moves above f_best
      - cost normalizes the total benefit by evaluation cost
    """
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Compute z-score for EI calculation
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: reward high uncertainty in regions near or below f_best
    # The exponential decay ensures we don't explore regions far above f_best
    alpha = 0.15
    beta = 0.8
    exploration_bonus = alpha * sigma_safe * np.exp(-beta * np.maximum(mu - f_best, 0.0))
    
    # Combine EI and exploration bonus, then normalize by cost
    combined_score = ei + exploration_bonus
    score = combined_score / cost_safe
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
