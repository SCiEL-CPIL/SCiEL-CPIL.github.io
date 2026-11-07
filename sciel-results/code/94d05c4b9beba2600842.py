import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a hybrid approach: Probability of Improvement (PI) combined with
    uncertainty-weighted exploration, normalized by evaluation cost.
    
    The key insight is that in short-budget scenarios, we want to:
    1. Exploit points likely to improve over f_best (PI component)
    2. Explore high-uncertainty regions where the GP is uncertain (sigma component)
    3. Prefer cheaper evaluations (cost normalization)
    
    We use a weighted combination that balances exploitation and exploration,
    with the balance shifting based on the current f_best value (as a proxy
    for how far we are from the optimum).
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute z-score for Probability of Improvement
    z = (f_best - mu) / sigma_safe
    
    # PI: probability that f(x) < f_best
    pi = norm.cdf(z)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    ei = (f_best - mu) * pi + sigma_safe * phi_z
    ei = np.maximum(0, ei)
    
    # Uncertainty-based exploration: higher sigma means more uncertain
    # Normalize sigma to [0, 1] range relative to observed values
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    sigma_norm = sigma_safe / (sigma_max + 1e-8)
    
    # Adaptive balance: as f_best approaches 0 (optimum), shift more to exploitation
    # Use a sigmoid-like transition based on f_best
    # When f_best is large (far from optimum), explore more; when small, exploit more
    f_best_abs = abs(f_best)
    # Transition parameter: beta controls how quickly we shift from exploration to exploitation
    beta = 2.0
    exploit_weight = 1.0 / (1.0 + np.exp(-beta * (f_best_abs - 0.1)))
    
    # Combine EI and exploration signal
    # Normalize EI to be comparable with sigma_norm
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / (ei_max + 1e-8)
    
    # Weighted combination
    combined = exploit_weight * ei_norm + (1 - exploit_weight) * sigma_norm
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    # But don't let cost dominate completely - use a power less than 1
    cost_penalty = np.power(cost_safe, 0.5)
    scores = combined / cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
