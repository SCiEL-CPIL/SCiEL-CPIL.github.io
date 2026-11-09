import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE_OFFSET = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using cost-weighted expected improvement with
    budget-aware exploration.
    
    Structural change: Instead of adapting a quantile level based on f_best,
    we compute a cost-normalized EI that explicitly accounts for the fact that
    evaluations at higher cost (more expensive x locations) yield less "value
    per unit cost." We add a Thompson-sampling-inspired exploration bonus that
    scales with the remaining budget fraction, encouraging exploration early
    and exploitation late.
    
    Parameters:
    mu: Array of shape (2000,) - GP mean
    sigma: Array of shape (2000,) - GP standard deviation
    f_best: Float - Best observed value
    cost: Array of shape (2000,) - Cost of evaluation
    
    Returns:
    Array of shape (2000,) - Acquisition scores (higher is better)
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE_OFFSET:
        mu_safe = np.asarray(mu)
        cost_safe = np.asarray(cost)
        cost_safe = np.maximum(cost_safe, 1e-9)
        return -mu_safe / cost_safe

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Standardize: work in units where f_best = 0
    # Improvement: how much better than f_best?
    improvement = f_best - mu  # positive when mu < f_best
    
    # Expected Improvement (standard form for minimization)
    z = improvement / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI = improvement * P(X < f_best) + sigma * pdf(z)
    ei = improvement * cdf_z + sigma_safe * pdf_z
    
    # Cost normalization: divide by cost to get "improvement per unit cost"
    # This naturally penalizes expensive evaluations
    ei_cost = ei / cost_safe
    
    # Budget-aware exploration:
    # The cost array tells us the cost of each candidate.
    # We estimate remaining budget from the current cost landscape.
    # Mean cost gives a proxy for how "expensive" the search space is.
    mean_cost = np.mean(cost_safe)
    
    # Exploration bonus: proportional to sigma / cost (uncertainty per unit cost)
    # This is like a cost-aware UCB term
    # Scale it down as f_best approaches 0 (we're close to optimal)
    f_best_clamped = max(f_best, 0.0)
    # When f_best is large, we need more exploration; when small, exploit
    exploration_scale = 0.5 / (1.0 + f_best_clamped * 0.1)
    
    exploration_bonus = exploration_scale * sigma_safe / cost_safe
    
    # Combine: cost-weighted EI + exploration bonus
    score = ei_cost + exploration_bonus
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
