import numpy as np
import time

_start_time = None
_call_count = 0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function for Bayesian optimization
    on Shekel 4D. Uses EI per unit cost as the scoring rule with a small UCB exploration
    term for robustness.
    """
    global _start_time, _call_count

    if _start_time is None:
        _start_time = time.time()
    _call_count += 1

    elapsed = time.time() - _start_time
    if elapsed > 310:
        # Safety fallback: simple cost-aware exploitation
        return -mu / np.maximum(cost, 1e-9)

    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-12, 1e6)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu(x), sigma(x)^2)
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # Standardize
    z = (f_best - mu) / sigma
    
    # Clip z to avoid extreme values causing numerical issues
    z = np.clip(z, -10, 10)
    
    # Phi and phi are the CDF and PDF of the standard normal
    # Use scipy.stats.norm for accuracy
    from scipy.stats import norm
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Only positive when z > 0 (i.e., mu < f_best)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    EI = (f_best - mu) * Phi_z + sigma * phi_z
    
    # EI is always non-negative; when mu >= f_best, EI is approximately 0
    
    # Cost-aware score: EI / cost (higher is better)
    # Add a small UCB term for exploration
    # UCB = mu - kappa * sigma (for minimization, lower is better, so we subtract)
    # But we want to maximize, so score = -mu + kappa*sigma, divided by cost
    # Let's blend EI/cost with a small exploration bonus
    
    # The EI formulation already balances exploration and exploitation.
    # Add a tiny UCB component to help escape flat regions
    kappa = 0.1
    ucb_score = (-mu + kappa * sigma) / cost_safe
    
    # Primary score is EI/cost, with a small UCB blend
    # In early iterations, EI may be small, so UCB helps explore
    if _call_count <= 10:
        alpha = 0.3  # more UCB influence early
    elif _call_count <= 30:
        alpha = 0.1
    else:
        alpha = 0.05
    
    scores = (1 - alpha) * (EI / cost_safe) + alpha * ucb_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
