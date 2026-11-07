"""Cost-aware acquisition for BO on four benchmark functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 285.0
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # delta = f_best - mu (potential improvement for minimization)
    delta = f_best - mu
    
    # Cost-aware Expected Improvement (EI)
    # EI = delta * Phi(z) + sigma * phi(z)
    # where z = delta / sigma
    z = delta / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = delta * cdf + sigma_safe * pdf
    
    # Cost-aware normalization
    cost_safe = np.maximum(cost, 1e-6)
    
    # Primary score: cost-aware EI
    score_ei = ei / cost_safe
    
    # Secondary score: cost-aware UCB for exploration
    # UCB = f_best - mu + kappa * sigma
    # We want to explore areas with high sigma that are also cheap
    
    # Adaptive kappa based on the current state
    # Early on (low total cost), explore more; later, exploit more
    # Use the mean cost as a proxy for how far along we are
    mean_cost = np.mean(cost)
    
    # When mean cost is low (early), use higher kappa for exploration
    # When mean cost is high (late), use lower kappa for exploitation
    # Scale kappa between 0.5 and 3.0 based on mean_cost
    # Assuming cost ranges roughly from 1 to 2 (since cost = 1 + mean(x_scaled))
    # So mean_cost ranges from ~1.2 to ~1.8
    # Normalize: (mean_cost - 1.0) / 1.0 gives 0.2 to 0.8
    # Map to kappa: 3.0 - 2.5 * normalized
    normalized_cost = np.clip((mean_cost - 1.0) / 1.0, 0, 1)
    kappa = 3.0 - 2.5 * normalized_cost
    
    ucb_term = f_best - mu + kappa * sigma_safe
    score_ucb = ucb_term / cost_safe
    
    # Combine EI and UCB scores
    # Use a weighted combination with adaptive weights
    # Early on, favor UCB more; later, favor EI more
    w_ucb = 0.6 - 0.3 * normalized_cost  # 0.6 to 0.3
    w_ei = 1.0 - w_ucb  # 0.4 to 0.7
    
    # Combined score
    scores = w_ei * score_ei + w_ucb * score_ucb
    
    # Normalize to ensure numerical stability
    scores_mean = np.mean(scores)
    scores_std = np.std(scores)
    if scores_std > 1e-10:
        scores = (scores - scores_mean) / scores_std
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return np.zeros_like(mu)
    
    return scores
# EVOLVE-BLOCK-END
