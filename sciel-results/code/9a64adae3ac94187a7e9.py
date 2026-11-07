import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Blended Cost-Aware Acquisition: Expected Improvement + UCB Exploration.
    
    Strategy:
    1. Compute standard Expected Improvement (EI) to capture probabilistic improvement.
    2. Compute Upper Confidence Bound (UCB) term for exploration: -mu + beta * sigma.
       Note: For minimization, we want to maximize (f_best - mu) + beta * sigma.
       The UCB score here is defined as the potential improvement plus exploration bonus.
    3. Blend EI and UCB scores with alpha = 0.4 (EI) and (1-alpha) = 0.6 (UCB).
       This leverages UCB's stronger exploration signal while retaining EI's dampening
       in regions where mu is already near f_best.
    4. Apply cost-aware normalization. We use a simple division by cost, which is
       robust and computationally efficient. The cost is 1 + mean(x_scaled), so
       cheaper points (lower x values) get higher scores.
    5. Return the blended, cost-normalized scores.
    
    Structural Change:
    - Combines the probabilistic improvement of EI with the aggressive exploration
      of UCB, rather than relying on a single criterion.
    - Uses a fixed blend ratio (0.4/0.6) tuned from observed improvements.
    """
    t0 = time.time()
    deadline = t0 + 290.0  # Safety margin before 300s limit
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    cost = np.asarray(cost).flatten()
    
    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # 1. Compute Expected Improvement (EI)
    # EI = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # 2. Compute UCB-style exploration score
    # For minimization, we want to maximize (f_best - mu) + beta * sigma
    # This is equivalent to the potential improvement plus an exploration bonus.
    beta = 3.0
    ucb_score = (f_best - mu) + beta * sigma_safe
    
    # 3. Blend EI and UCB scores
    # Alpha controls the weight of EI. Higher alpha = more exploitation.
    # Lower alpha = more exploration (UCB dominates).
    # Observed improvement with alpha=0.4 (EI) and 0.6 (UCB).
    alpha = 0.4
    blended_score = alpha * ei + (1.0 - alpha) * ucb_score
    
    # 4. Cost-aware normalization
    # Divide by cost to prefer cheaper evaluations.
    # cost = 1 + mean(x_scaled), so cost in [1, 2].
    scores = blended_score / cost_safe
    
    # 5. Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline check
    if time.time() > deadline:
        # Fallback: return simple EI/cost if we're close to timeout
        fallback_scores = ei / cost_safe
        return np.nan_to_num(fallback_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
