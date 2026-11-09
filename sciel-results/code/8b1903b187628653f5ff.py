import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE_OFFSET = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
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
        # Fallback: Exploit the best mean, adjusted by cost
        mu_safe = np.asarray(mu)
        cost_safe = np.asarray(cost)
        cost_safe = np.maximum(cost_safe, 1e-9)
        return -mu_safe / cost_safe

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Ensure sigma and cost are positive to avoid division errors
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Expected Improvement (EI)
    # EI = E[max(0, f_best - f(x))]
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # 2. Adaptive Cost Penalty
    # The global minimum of the shifted Rastrigin is in the inner 80% of the domain.
    # The cost is 1 + mean(x_scaled).
    # The center (x=0) has cost 1.5. The corners have costs 1 and 2.
    # We want to avoid being too biased towards the cheap corners if the minimum is in the middle.
    # However, as we converge (f_best decreases), we should be more sensitive to cost to save budget.
    
    # Dynamic exponent for cost penalty:
    # When f_best is high (early search), exponent is close to 1 (standard 1/cost).
    # When f_best is low (late search), exponent increases (stronger penalty for expensive points).
    # This helps escape local minima early (where cost matters less) and fine-tune later (where saving budget for precision matters).
    
    # f_best is the observed minimum. For Rastrigin, min is 0.
    # If f_best is close to 0, we are near the global min.
    # We use a smooth transition.
    # exponent = 1.0 + alpha / (f_best + epsilon)
    # If f_best is 10, exponent ~ 1.01.
    # If f_best is 0.1, exponent ~ 1.0 + 1/0.1 = 11? Too high.
    # Let's cap the effect.
    # Use: exponent = 1.0 + 1.0 / (1.0 + f_best)
    # If f_best = 10, exp = 1.09.
    # If f_best = 0, exp = 2.0.
    # This seems reasonable.
    
    exponent = 1.0 + 1.0 / (1.0 + np.maximum(f_best, 0.0))
    
    # 3. Compute Score
    # score = EI / (cost ** exponent)
    # This is equivalent to EI * cost ** (-exponent)
    
    # To avoid numerical issues with large exponents on small costs, we can compute log-score if needed,
    # but cost is in [1, 2], so cost ** 2 is at most 4. No overflow risk.
    
    score = ei / (cost_safe ** exponent)
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
