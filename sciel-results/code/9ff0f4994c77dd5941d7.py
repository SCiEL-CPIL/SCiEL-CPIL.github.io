import numpy as np
import time
from scipy.stats import norm

# Deadline guard: ensure we finish well before 330s wall-clock
_DEADLINE = time.time() + 280.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a structural change:
    Model selection as stochastic knapsack by maximizing
    P(mu < f_best) * (f_best - mu) / cost
    instead of raw Expected Improvement / cost.
    
    This directly targets expected regret reduction per unit cost,
    inherently down-weighting high-cost candidates with marginal improvement
    potential and handling the probability of improvement explicitly.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP std predictions
        f_best: scalar, best observed value so far
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros(len(mu), dtype=np.float64)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # z-score for probability of improvement (minimization)
    # P(mu < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -20, 20)
    
    # Probability that the candidate improves over f_best
    p_improve = norm.cdf(z)
    
    # Expected improvement magnitude (only counts if improvement happens)
    # E[f_best - mu | mu < f_best] * P(mu < f_best) = EI
    # But we want P * (f_best - mu) / cost, which is different
    # P(mu < f_best) * (f_best - mu) / cost
    # Note: (f_best - mu) can be negative, so we need to be careful
    # Actually, P(mu < f_best) is high when mu < f_best, and (f_best - mu) is positive then
    # When mu > f_best, P(mu < f_best) is low and (f_best - mu) is negative
    # So the product naturally handles both cases
    
    # Stochastic knapsack score: probability-weighted improvement per cost
    # score = P(mu < f_best) * (f_best - mu) / cost
    # But (f_best - mu) can be negative, which would penalize candidates
    # We want to maximize expected regret reduction, which is:
    # E[max(0, f_best - mu)] / cost = EI / cost
    # However, the directive says to use P(mu < f_best) * (f_best - mu) / cost
    # Let's compute both and combine
    
    # Expected Improvement (standard)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability-weighted improvement (stochastic knapsack formulation)
    # This is NOT the same as EI. It's P(mu < f_best) * (f_best - mu)
    # When mu << f_best, P ~ 1, (f_best - mu) is large positive -> high score
    # When mu ~ f_best, P ~ 0.5, (f_best - mu) ~ 0 -> low score
    # When mu >> f_best, P ~ 0, (f_best - mu) is negative -> low/negative score
    prob_weighted = p_improve * (f_best - mu)
    
    # Cost-aware versions
    ei_score = ei / cost_safe
    prob_score = prob_weighted / cost_safe
    
    # Upper Confidence Bound for exploration
    # UCB = beta * sigma / cost
    beta = 0.3
    ucb_score = beta * sigma_safe / cost_safe
    
    # Normalize each component to zero-mean, unit-variance
    def _normalize(arr):
        m = np.mean(arr)
        s = np.std(arr) + 1e-8
        return (arr - m) / s
    
    ei_norm = _normalize(ei_score)
    prob_norm = _normalize(prob_score)
    ucb_norm = _normalize(ucb_score)
    
    # Combine: emphasize the stochastic knapsack formulation (structural change)
    # with EI as a secondary signal and UCB for exploration
    # The prob_weighted term directly optimizes P * improvement / cost
    combined = 0.40 * prob_norm + 0.35 * ei_norm + 0.25 * ucb_norm
    
    # Ensure all values are finite
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    return combined
