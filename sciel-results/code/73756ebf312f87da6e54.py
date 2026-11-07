"""Cost-aware acquisition for rastrigin 2d BO loop."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
_T0 = time.time()
_DEADLINE = 320.0  # safety margin of 10s before the 330s kill


def _deadline_exceeded():
    return (time.time() - _T0) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Knowledge Gradient (KG) approximation
    for Bayesian optimization of rastrigin 2d.
    
    The KG estimates the expected reduction in the global minimum value.
    For multimodal functions like Rastrigin, this naturally prioritizes points
    that may reveal new basins (exploration) while still exploiting promising regions.
    
    We use an approximation: KG(x) ≈ E[max(0, f_best - f(x))] but adjusted to
    account for the probability that f(x) is a new global minimum given the GP posterior.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if _deadline_exceeded():
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei / (cost + 1e-9)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma and cost are positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score: z = (f_best - mu) / sigma
    # This is the number of standard deviations by which f(x) is expected to beat f_best
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Clip to avoid numerical issues
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Knowledge Gradient approximation:
    # The true KG is E[max(0, f_best - min(f(x), f_best_after))]
    # where f_best_after accounts for other points. A common approximation
    # is to use EI but with a correction that accounts for the "value of information"
    # from reducing uncertainty.
    
    # A practical KG approximation for BO:
    # KG(x) ≈ EI(x) * (1 - exp(-sigma(x)^2 / scale))
    # where scale is related to the problem's scale.
    
    # For rastrigin, the function values can be large. We use an adaptive scale.
    # When f_best is large, we want more exploration (higher sigma weight).
    # When f_best is small, we want more exploitation.
    
    # Alternative: use a weighted combination of EI and a "novelty" term
    # that rewards high uncertainty regions, but discounted by cost.
    
    # Let's use a simpler but effective approach:
    # score = EI(x) / cost(x) + lambda * sigma(x) / cost(x) * (1 - pi(x))
    # The second term rewards exploration of uncertain regions that are not
    # already likely to improve f_best.
    
    # Adaptive lambda based on f_best
    if f_best < 0.1:
        # Very close to optimum: exploit heavily
        lambda_explore = 0.1
    elif f_best < 1.0:
        lambda_explore = 0.3
    elif f_best < 10.0:
        lambda_explore = 0.5
    else:
        # Far from optimum: explore more
        lambda_explore = 0.8
    
    # Novelty term: high sigma, low probability of improvement
    novelty = sigma_safe * (1 - pi)
    
    # Normalize EI and novelty to be comparable
    # EI is in objective units, novelty is in objective units too
    f_scale = abs(f_best) + 1.0
    ei_normalized = ei / f_scale
    novelty_normalized = novelty / f_scale
    
    # Combine
    score = ei_normalized + lambda_explore * novelty_normalized
    
    # Cost-awareness: divide by cost
    score = score / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
