import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization.
    Combines Expected Improvement (EI) and Upper Confidence Bound (UCB)
    with z-score normalization to balance exploitation and exploration.
    """
    # Deadline guard
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    if elapsed > 325.0:  # Safety margin before 330s kill
        # Fallback: simple heuristic preferring low mu and low cost
        mu_fallback = np.asarray(mu, dtype=np.float64)
        cost_fallback = np.asarray(cost, dtype=np.float64)
        score = -mu_fallback / np.maximum(cost_fallback, 1e-9)
        return np.where(np.isfinite(score), score, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)

    # 1. Expected Improvement (EI) for minimization
    # Improvement is f_best - f(x). If f(x) < f_best, improvement is positive.
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in cdf/pdf for extreme values
    z_clipped = np.clip(z, -50.0, 50.0)
    
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI score
    score_ei = ei / cost_safe

    # 2. Upper Confidence Bound (UCB) for minimization
    # We want to maximize the potential for a lower value.
    # Optimistic lower bound: L(x) = mu(x) - beta * sigma(x).
    # We want to select points where L(x) is very low (highly negative).
    # So we maximize -L(x) = -mu(x) + beta * sigma(x).
    beta = 2.0
    ucb_score = -mu + beta * sigma_safe
    
    # Cost-aware UCB score
    score_ucb = ucb_score / cost_safe

    # 3. Z-score normalization to balance scales
    # This allows the exploration term (UCB) to compete with exploitation (EI)
    def zscore(arr):
        m = np.mean(arr)
        s = np.std(arr)
        if s < 1e-9:
            return np.zeros_like(arr)
        return (arr - m) / s
        
    z_ei = zscore(score_ei)
    z_ucb = zscore(score_ucb)
    
    # 4. Combine
    # 75% weight to exploitation (EI), 25% to exploration (UCB)
    final_score = 0.75 * z_ei + 0.25 * z_ucb
    
    # Ensure finite values
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
