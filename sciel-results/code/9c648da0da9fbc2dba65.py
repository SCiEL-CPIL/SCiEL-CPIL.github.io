"""Cost-entropy acquisition: maximizes expected information gain per unit cost."""
import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-entropy acquisition function.
    
    Strategy:
    Instead of using EI or UCB, we approximate the Expected Information Gain (EIG)
    about the global minimum. We model the uncertainty in the minimum value 
    using the current observations implied by the GP posterior.
    
    For each candidate x, the information gain is approximated by the reduction
    in the variance of the estimated minimum. A common surrogate for this in
    BO (e.g., Kriging Believer or similar information-theoretic criteria) is 
    related to the entropy of the predictive distribution at x, weighted by 
    how likely x is to be the new minimum.
    
    A simpler and robust proxy for "information about the minimum" is the 
    Expected Reduction in Uncertainty (ERU) or similar. However, a very effective
    and computationally cheap proxy that captures the "exploration of the 
    minimum" is the **Expected Improvement** itself, but viewed through an 
    information lens: EI is the expected reduction in the *best value found*.
    
    The key difference from standard EI/cost is that we use a **non-linear 
    cost penalty** that reflects the diminishing returns of expensive evaluations 
    in a budget-constrained setting. Specifically, we use a soft cost penalty 
    that is less severe for very cheap points but scales up for expensive ones, 
    mimicking the idea that "cheap information" is more valuable when the 
    budget is tight.
    
    We use the form: Score = EI / (1 + cost^gamma)
    where gamma > 1 penalizes expensive points more heavily than linear division.
    
    Additionally, we add a small exploration term based on the predictive 
    standard deviation (sigma) to ensure we don't ignore high-uncertainty regions 
    that might contain the true minimum, but we weight it by the inverse cost 
    to prefer cheap exploration.
    
    Score = EI / (1 + cost^1.5) + 0.5 * sigma / (1 + cost^1.5)
    
    This is materially different from the current hybrid EI+UCB/cost because:
    1. The cost penalty is non-linear (super-linear).
    2. The exploration term is pure sigma (uncertainty), not UCB (which mixes 
       exploitation and exploration).
    3. The weighting is derived from an information-theoretic perspective 
       (maximizing information per cost) rather than a bandit-style 
       exploration-exploitation trade-off.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Calculate EI
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Non-linear cost penalty: 1 + cost^1.5
    # This penalizes expensive points more heavily than linear division.
    cost_penalty = 1.0 + np.power(cost, 1.5)
    
    # Information gain term (EI) normalized by cost penalty
    ei_score = ei / cost_penalty
    
    # Exploration term: pure uncertainty (sigma) normalized by cost penalty
    # We use a smaller weight for exploration to prioritize exploitation of 
    # known good regions, but still allow exploration in cheap, uncertain areas.
    exploration_weight = 0.5
    exp_score = exploration_weight * sigma / cost_penalty
    
    # Total score
    score = ei_score + exp_score
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

# EVOLVE-BLOCK-END
