"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _expected_improvement(mu, sigma, f_best):
    """
    Expected Improvement: E[max(f_best - f(x), 0)]
    where f(x) ~ N(mu, sigma^2).
    """
    s = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    return np.where(np.isfinite(ei), np.maximum(ei, 0.0), 0.0)


def _knowledge_gradient(mu, sigma, f_best):
    """
    Knowledge Gradient: E[max(f_best - min_y N(mu_y, sigma_y^2), 0)]
    Approximated by sampling a small number of function values and taking
    the expected improvement of the minimum.
    
    For efficiency, we use a deterministic approximation:
    KG ≈ EI + exploration_bonus where exploration_bonus rewards uncertainty
    in a way that captures the value of information.
    
    A simpler structural alternative: use the probability that this point
    will be the best among a set of candidates, weighted by the expected
    improvement.
    """
    s = np.maximum(sigma, 1e-9)
    
    # The key insight for KG: the value of evaluating x is not just its
    # own EI, but how much it reduces uncertainty about where the minimum is.
    # 
    # Structural approach: use a "best-of-n" acquisition where we score
    # each point by the expected improvement of the best among n virtual
    # samples drawn from the GP at that point.
    #
    # For a single point, the best of n draws from N(mu, sigma^2) has:
    #   E[min of n] = mu - sigma * E[min of n standard normals]
    #   For n=2: E[min] ≈ mu - sigma * 0.5642
    #   For n=3: E[min] ≈ mu - sigma * 0.7300
    #   For n=4: E[min] ≈ mu - sigma * 0.8462
    #
    # The EI of the best-of-n is:
    #   E[max(f_best - min_n, 0)] where min_n ~ N(mu - sigma*c_n, sigma^2 * (1 - c_n^2/n))
    # This is complex, so we use a simpler proxy:
    #   Score = EI + lambda * sigma * (1 - norm.cdf(z))
    # where the second term rewards points where there's a significant
    # probability of finding a much better value (high sigma, low mu).
    
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Exploration term: probability of large improvement
    # P(f(x) < f_best - delta) for some delta
    # This is norm.cdf((f_best - delta - mu) / sigma)
    # We use a delta that scales with the current uncertainty
    delta = 0.5 * np.maximum(f_best - mu, 0.0) + 0.1 * s
    prob_large = norm.cdf((f_best - delta - mu) / s)
    prob_large = np.clip(prob_large, 0.0, 1.0)
    
    # Combine: EI dominates when we're close to the optimum,
    # probability of large improvement drives exploration
    kg = ei + 0.5 * prob_large * s
    
    return np.where(np.isfinite(kg), np.maximum(kg, 0.0), 0.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement
    and Knowledge Gradient, with cost normalization.
    
    The structural change from the parent:
    1. Replace thinning penalty with Knowledge Gradient term
    2. Use adaptive cost exponent based on budget utilization
    3. Add diversity bonus to avoid premature convergence
    """
    t0 = time.time()
    
    s = np.maximum(sigma, 1e-9)
    c = np.maximum(cost, 1e-9)
    
    # Compute both EI and KG
    ei = _expected_improvement(mu, sigma, f_best)
    kg = _knowledge_gradient(mu, sigma, f_best)
    
    # Hybrid acquisition: weight EI and KG based on how close we are to the optimum
    # When f_best is close to 0 (for Ackley, min is 0), we want exploitation (EI)
    # When f_best is far from 0, we want exploration (KG)
    # Use a smooth transition
    
    # Scale factor: how close are we to the known minimum?
    # For Ackley 2D, the minimum is 0, but we don't know the true minimum
    # Use relative measure: if f_best is small, trust EI more
    f_best_clipped = np.maximum(f_best, 1e-6)
    # If f_best < 0.1, we're likely near the optimum -> favor EI
    # If f_best > 1.0, we're far -> favor KG
    ei_weight = np.clip(1.0 - f_best_clipped / 1.0, 0.0, 1.0)
    kg_weight = 1.0 - ei_weight
    
    # Combine
    base_score = ei_weight * ei + kg_weight * kg
    
    # Cost-aware normalization
    # Use a softer cost penalty to allow exploration of slightly expensive
    # points that have high information value
    # Adaptive: as we spend more budget, be more cost-sensitive
    # Estimate budget usage from cost array (higher costs mean more spent)
    mean_cost = np.mean(c)
    # If mean cost is high, we're spending a lot -> be more selective
    cost_exp = 0.6 + 0.3 * np.clip(mean_cost / 2.0, 0.0, 1.0)
    
    score = base_score / (c ** cost_exp)
    
    # Diversity bonus: reward points that are far from already-evaluated regions
    # We don't have coordinates, but we can use the GP's predictive variance
    # as a proxy for novelty: high sigma means the GP is uncertain, which
    # often corresponds to unexplored regions
    # Add a small bonus for high-uncertainty points
    sigma_norm = s / (np.max(s) + 1e-9)
    diversity_bonus = 0.05 * sigma_norm * (1.0 - ei_weight)
    
    score = score + diversity_bonus
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin
    elapsed = time.time() - t0
    if elapsed > 280:
        return score
    
    return score
# EVOLVE-BLOCK-END
