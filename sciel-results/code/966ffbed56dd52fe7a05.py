"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Knowledge Gradient (KG) acquisition function with cost-aware discount.
    
    KG estimates the expected improvement in the global minimum after observing
    a new point. We approximate this by:
    1. For each candidate x, compute the probability that f(x) < f_best
    2. Estimate the expected new minimum as a weighted combination
    3. Scale by 1/sqrt(cost) for cost-awareness (sublinear discount)
    """
    # Ensure sigma is non-negative with a floor
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-scores: how many sigmas below f_best each candidate's mean is
    z = (f_best - mu) / sigma
    
    # P(f(x) < f_best) = Phi(z)
    p_improve = norm.cdf(z)
    
    # Expected value of f(x) given that f(x) < f_best (truncated normal mean)
    # E[f(x) | f(x) < f_best] = mu - sigma * phi(z) / Phi(z) when Phi(z) > 0
    # But we need to handle cases where Phi(z) is very small
    
    # Expected minimum if we evaluate x:
    # new_min = min(f_best, f(x))
    # E[new_min] = f_best * P(f(x) >= f_best) + E[f(x) | f(x) < f_best] * P(f(x) < f_best)
    #            = f_best * (1 - p_improve) + (mu - sigma * phi(z)/Phi(z)) * p_improve
    
    # KG = f_best - E[new_min] = expected improvement in the global minimum
    #    = f_best - [f_best * (1 - p) + (mu - sigma*phi(z)/Phi(z)) * p]
    #    = f_best * p - (mu - sigma*phi(z)/Phi(z)) * p
    #    = p * (f_best - mu + sigma*phi(z)/Phi(z))
    #    = p * (f_best - mu) + sigma * phi(z)
    #    = EI (standard expected improvement!)
    
    # So standard EI is actually a valid KG approximation when we only consider
    # the current best. For a true KG, we'd need to consider all previous points,
    # but with 2000 candidates and no access to previous points, we use EI as 
    # the KG proxy.
    
    # To differentiate from plain EI, we add an exploration bonus that accounts
    # for the potential to discover a much better minimum in high-uncertainty regions.
    # This is inspired by the idea that high-uncertainty regions may contain the
    # true global minimum, so we weight them more.
    
    ei = (f_best - mu) * p_improve + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Exploration bonus: reward high-uncertainty regions that are also promising
    # (i.e., not too far above f_best). We use a Gaussian kernel on the z-score
    # to focus on regions where improvement is plausible but not certain.
    # The optimal z for exploration is around 0.5-1.0 (moderate uncertainty, 
    # reasonable chance of improvement)
    exploration_bonus = sigma * np.exp(-0.5 * (z - 0.8) ** 2)
    
    # Combine EI and exploration bonus
    # Weight exploration more when EI is low (to explore) and less when EI is high
    max_ei = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_weight = np.minimum(ei / (max_ei * 0.5), 1.0)
    
    benefit = (1 - 0.3 * ei_weight) * ei + 0.3 * ei_weight * ei + (1 - ei_weight) * 0.5 * exploration_bonus
    # Simplified: benefit = ei + 0.5 * (1 - ei_weight) * exploration_bonus
    benefit = ei + 0.5 * (1 - ei_weight) * exploration_bonus
    
    # Cost-aware discount: sublinear 1/sqrt(cost) to balance cost vs. benefit
    # This is less aggressive than 1/cost, allowing some exploration of 
    # higher-cost regions if they offer significantly higher benefit
    cost_discount = 1.0 / np.sqrt(cost)
    
    # Final score
    score = benefit * cost_discount
    
    # Ensure all scores are finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
