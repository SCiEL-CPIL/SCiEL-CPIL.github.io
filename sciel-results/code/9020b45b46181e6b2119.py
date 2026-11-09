"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
_DEADLINE_START = time.time()


def _expected_improvement_cost(mu, sigma, f_best, cost, cost_exp=1.5):
    """
    Cost-aware Expected Improvement for minimization.
    
    For minimization, improvement is (f_best - f(x)).
    EI = E[max(f_best - f(x), 0)]
    
    We divide by cost^cost_exp to get expected improvement per unit cost.
    Using cost_exp > 1 penalizes expensive points more strongly.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    # For minimization: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Numerical stability
    
    # EI formula for minimization
    # E[max(f_best - f, 0)] where f ~ N(mu, sigma^2)
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # Clip to non-negative
    
    # Cost-adjusted: divide by cost^cost_exp
    cost_safe = np.maximum(cost, 1e-6)
    return ei / (cost_safe ** cost_exp)


def _lower_confidence_bound_cost(mu, sigma, f_best, cost, xi=0.0, kappa=1.0, cost_exp=1.5):
    """
    Cost-aware Lower Confidence Bound (for minimization).
    
    LCB = mu - kappa * sigma  (lower is better for minimization)
    We want to maximize negative LCB per unit cost.
    """
    lcb = mu - kappa * sigma
    cost_safe = np.maximum(cost, 1e-6)
    # Higher score = better: use -lcb / cost^cost_exp
    return -lcb / (cost_safe ** cost_exp)


def _probability_of_improvement_cost(mu, sigma, f_best, cost, cost_exp=1.5):
    """
    Cost-aware Probability of Improvement for minimization.
    
    PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    pi = norm.cdf(z)
    cost_safe = np.maximum(cost, 1e-6)
    return pi / (cost_safe ** cost_exp)


def _hypervolume_contribution(cost_arr, ei_arr, ref_cost=None, ref_ei=None):
    """
    Compute hypervolume contribution of each point in a 2D objective space.
    
    We minimize cost and maximize EI, so we minimize (cost, -EI).
    Points: (N, 2) array where each row is (cost, -EI).
    
    Returns: (N,) array of hypervolume contributions.
    """
    n = len(cost_arr)
    if n == 0:
        return np.array([])
    
    # Build the 2D objective: minimize (cost, -EI)
    obj = np.column_stack([cost_arr, -ei_arr])
    
    # Reference point: worst acceptable values
    if ref_cost is None:
        ref_cost = np.max(cost_arr) + 0.1
    if ref_ei is None:
        ref_ei = 0.0  # -EI = 0 means EI = 0, which is the worst
    
    ref_point = np.array([ref_cost, ref_ei])
    
    # Sort by cost (first objective, minimize)
    order = np.argsort(obj[:, 0])
    sorted_obj = obj[order]
    
    contributions = np.zeros(n)
    
    # For each point in sorted order, compute its unique hypervolume contribution
    # A point's contribution is the area of the rectangle it uniquely dominates
    # relative to the reference point, minus areas dominated by other points.
    
    # For 2D, we can compute this efficiently:
    # Sort by first objective. For point i, find the next point j > i that has
    # a smaller second objective (i.e., higher EI). The width is the cost difference,
    # and the height is the difference in -EI from this point to the reference.
    
    # Precompute minimum -EI from each position to the end
    min_neg_ei_from_end = np.minimum.accumulate(sorted_obj[:, 1])[::-1]
    
    for i in range(n - 1):
        # Find the first subsequent point with -EI <= current -EI
        current_neg_ei = sorted_obj[i, 1]
        
        # Look for next point with smaller or equal -EI
        subsequent_neg_ei = sorted_obj[i+1:, 1]
        mask = subsequent_neg_ei <= current_neg_ei
        
        if np.any(mask):
            next_idx = i + 1 + np.argmax(mask)
            cost_upper = sorted_obj[next_idx, 0]
        else:
            cost_upper = ref_cost
        
        cost_width = cost_upper - sorted_obj[i, 0]
        if cost_width <= 0:
            contributions[i] = 0
            continue
        
        # Height: from current -EI to reference -EI
        ei_height = ref_ei - current_neg_ei
        if ei_height <= 0:
            contributions[i] = 0
            continue
        
        contributions[i] = cost_width * ei_height
    
    # Last point
    i = n - 1
    cost_width = ref_cost - sorted_obj[i, 0]
    ei_height = ref_ei - sorted_obj[i, 1]
    if cost_width > 0 and ei_height > 0:
        contributions[i] = cost_width * ei_height
    
    # Map back to original order
    result = np.zeros(n)
    result[order] = contributions
    return result


def _pareto_hypervolume_acquisition(mu, sigma, f_best, cost):
    """
    Pareto-frontier hypervolume contribution acquisition.
    
    This explicitly models the multi-objective trade-off between
    expected improvement and cost, selecting candidates that
    maximize their contribution to the hypervolume of the
    non-dominated (cost, -EI) frontier.
    """
    # Compute EI
    ei = _expected_improvement_cost(mu, sigma, f_best, cost, cost_exp=1.0)
    
    # Compute hypervolume contributions
    hv = _hypervolume_contribution(cost, ei)
    
    return hv


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    array of shape (2000,) - acquisition scores (higher = better)
    """
    # Safety: ensure inputs are finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Check time budget
    elapsed = time.time() - _DEADLINE_START
    if elapsed > 315:  # Safety margin
        # Simple fallback
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.nan_to_num(ei / (cost_safe ** 1.5), nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute multiple acquisition functions
    # 1. Cost-aware Expected Improvement with stronger cost penalty
    ei_score = _expected_improvement_cost(mu, sigma, f_best, cost, cost_exp=1.5)
    
    # 2. Cost-aware Lower Confidence Bound
    lcb_score = _lower_confidence_bound_cost(mu, sigma, f_best, cost, xi=0.0, kappa=1.0, cost_exp=1.5)
    
    # 3. Cost-aware Probability of Improvement
    pi_score = _probability_of_improvement_cost(mu, sigma, f_best, cost, cost_exp=1.5)
    
    # 4. Pareto hypervolume contribution
    hv_score = _pareto_hypervolume_acquisition(mu, sigma, f_best, cost)
    
    # Normalize each score to have zero mean and unit variance
    def normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        std = np.std(x)
        if std < 1e-10:
            return x
        return (x - np.mean(x)) / std
    
    ei_norm = normalize(ei_score)
    lcb_norm = normalize(lcb_score)
    pi_norm = normalize(pi_score)
    hv_norm = normalize(hv_score)
    
    # Adaptive weights based on optimization progress
    # f_best close to -0.8 means we're near the optimum
    # f_best far from -0.8 means we need more exploration
    min_possible = -0.8
    max_possible = 0.8  # rough upper bound for cosine8
    progress = 1.0 - min(max((f_best - min_possible) / (max_possible - min_possible), 0.0), 1.0)
    
    # Early on (high progress): more exploration (LCB, HV)
    # Later (low progress): more exploitation (EI, PI)
    w_ei = 0.30 + 0.15 * (1.0 - progress)
    w_lcb = 0.20 + 0.10 * progress
    w_pi = 0.15 + 0.05 * (1.0 - progress)
    w_hv = 0.35 + 0.05 * progress  # Give hypervolume a strong weight
    
    # Ensure weights sum to 1
    total_w = w_ei + w_lcb + w_pi + w_hv
    w_ei /= total_w
    w_lcb /= total_w
    w_pi /= total_w
    w_hv /= total_w
    
    # Combined score
    score = w_ei * ei_norm + w_lcb * lcb_norm + w_pi * pi_norm + w_hv * hv_norm
    
    # Add a small bonus for low-cost points to encourage efficient exploration
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = max(cost_max - cost_min, 1e-10)
    cost_bonus = 0.05 * (1.0 - (cost_safe - cost_min) / cost_range)
    
    score = score + cost_bonus
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Final time check
    if time.time() - _DEADLINE_START > 320:
        # Emergency fallback
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.nan_to_num(ei / (cost_safe ** 1.5), nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
