import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 315.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement
    and Upper Confidence Bound, adapted for the Styblinski-Tang 2D problem.
    
    The acquisition function balances:
    1. Exploitation: Points with low predicted mean (mu)
    2. Exploration: Points with high uncertainty (sigma)
    3. Cost awareness: Prefer points with lower evaluation cost
    
    We use a cost-adjusted EI that incorporates the cost into the improvement
    calculation, plus a UCB term for exploration.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback to simple cost-aware mean minimization
        cost_safe = np.where(np.isfinite(cost) & (cost > 1e-6), cost, 1.0)
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        return -mu_safe / cost_safe

    # Sanitize inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-6)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost) & (cost > 1e-6), cost, 1.0)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Compute cost-adjusted Expected Improvement
    # Standard EI: E[max(f_best - f(x), 0)]
    # For minimization: EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # Normalize by sigma for the normal CDF/PDF
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI can be negative if mu > f_best, clip to 0
    ei = np.maximum(ei, 0)
    
    # Compute UCB for exploration
    # Adaptive beta based on the number of evaluations (approximated by cost sum)
    total_cost = np.sum(cost)
    # As we gather more data, reduce exploration
    beta = 2.0 / (1.0 + np.log1p(total_cost / 10.0))
    
    ucb = -mu + beta * sigma
    
    # Combine EI and UCB with cost awareness
    # Weight EI more heavily as we converge
    # Use a blending parameter that increases with total cost
    blend = np.clip(total_cost / 50.0, 0.3, 0.9)
    
    # Cost-aware score: divide by cost to prefer cheap points
    # But also add a small bonus for high uncertainty to ensure exploration
    cost_factor = 1.0 / cost
    
    # Primary score: cost-adjusted EI
    score_ei = ei * cost_factor
    
    # Secondary score: cost-adjusted UCB
    score_ucb = ucb * cost_factor
    
    # Blend the two
    score = blend * score_ei + (1 - blend) * score_ucb
    
    # Add a small exploration bonus for points with high uncertainty
    # This helps escape local optima
    sigma_max = np.max(sigma)
    if sigma_max > 1e-6:
        exploration_bonus = 0.1 * (sigma / sigma_max)
        score = score + exploration_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -1e6)
    
    # Normalize scores to [0, 1] for consistency
    score_min = np.min(score)
    score_max = np.max(score)
    
    if score_max > score_min + 1e-10:
        score = (score - score_min) / (score_max - score_min)
    else:
        score = np.zeros_like(score)
    
    return score
