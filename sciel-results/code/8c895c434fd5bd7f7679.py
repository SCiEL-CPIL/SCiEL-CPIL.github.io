"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for the session
_SESSION_START = time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Quantile of the Posterior (QoP) acquisition function.
    
    For minimization, we use the p-th quantile of the posterior to target
    high-confidence low-value regions. The p-th quantile is:
        q_p = mu + sigma * norm.ppf(p)
    For minimization, we want to MINIMIZE q_p (find points where even the
    p-th quantile is low), so the acquisition score is -q_p / cost.
    
    We blend QoP with a UCB-like exploration term to navigate Rastrigin's
    multimodal landscape, and divide by cost to penalize expensive evaluations.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate, shape (2000,).
        
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Safety deadline guard: if we've used > 290s of the 330s limit,
    # return a simple fallback immediately.
    elapsed = time.time() - _SESSION_START
    if elapsed > 290:
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        # Simple QoP fallback with p=0.25
        p = 0.25
        z_p = norm.ppf(p)  # ~ -0.675
        qop = mu + sigma_safe * z_p
        score = -qop / cost_safe
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Quantile of the Posterior (QoP) for minimization
    # Use p=0.25 (25th percentile) as a balance between exploitation and exploration
    # q_0.25 = mu + sigma * norm.ppf(0.25) = mu - 0.675*sigma
    # We want to minimize q_0.25, so score = -q_0.25 / cost
    p = 0.25
    z_p = norm.ppf(p)  # Approximately -0.6745
    qop = mu + sigma_safe * z_p
    qop_score = -qop / cost_safe
    
    # UCB-like exploration term for additional exploration in uncertain regions
    # This helps navigate Rastrigin's multimodal landscape
    kappa = 1.0
    ucb_improvement = -mu + kappa * sigma_safe
    ucb_score = ucb_improvement / cost_safe
    
    # EI for exploitation of known good regions
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei_score = ei / cost_safe
    
    # Blend: 50% QoP, 25% UCB, 25% EI
    # QoP provides the main signal, UCB adds exploration, EI adds exploitation
    alpha_qop = 0.5
    alpha_ucb = 0.25
    alpha_ei = 0.25
    score = alpha_qop * qop_score + alpha_ucb * ucb_score + alpha_ei * ei_score
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
