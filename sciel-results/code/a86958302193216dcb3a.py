# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 300.0  # seconds, with safety margin


def _expected_improvement(mu, sigma, f_best):
    """Compute Expected Improvement (EI) for minimization."""
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Where sigma is very small, EI should be close to max(f_best - mu, 0)
    small_sigma = sigma < eps * 10
    ei[small_sigma] = np.maximum(f_best[small_sigma] - mu[small_sigma], 0.0)
    return np.maximum(ei, 0.0)


def _expected_improvement_lower_confidence(mu, sigma, f_best, xi=0.01):
    """Compute EI with a small exploration parameter xi."""
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - xi - mu) / sigma_safe
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    small_sigma = sigma < eps * 10
    ei[small_sigma] = np.maximum(f_best[small_sigma] - xi - mu[small_sigma], 0.0)
    return np.maximum(ei, 0.0)


def _ucb(mu, sigma, kappa=1.1):
    """Upper Confidence Bound for minimization (negative UCB)."""
    return mu + kappa * sigma


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: if we're too close to the limit, return simple EI/cost
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, eps)
        return np.where(np.isfinite(score), score, 0.0)
    
    eps = 1e-9
    cost_safe = np.maximum(cost, eps)
    
    # Compute multiple acquisition strategies
    # 1. Standard EI / cost
    ei = _expected_improvement(mu, sigma, f_best)
    score_ei = ei / cost_safe
    
    # 2. EI with exploration / cost
    ei_xi = _expected_improvement_lower_confidence(mu, sigma, f_best, xi=0.005)
    score_ei_xi = ei_xi / cost_safe
    
    # 3. UCB / cost (for exploration)
    ucb = _ucb(mu, sigma, kappa=1.0)
    score_ucb = -ucb / cost_safe  # negative because lower mu is better
    
    # 4. PI (Probability of Improvement) / cost
    z = (f_best - mu) / np.maximum(sigma, eps)
    pi = norm.cdf(z)
    score_pi = pi / cost_safe
    
    # 5. A blended approach: weight EI heavily, add some exploration
    # Use a soft-max-like combination to balance exploitation and exploration
    # Weight based on uncertainty: when sigma is high, favor UCB more
    
    # Normalize scores to [0, 1] range for blending
    def _normalize(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < eps:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    n_ei = _normalize(score_ei)
    n_ei_xi = _normalize(score_ei_xi)
    n_ucb = _normalize(score_ucb)
    n_pi = _normalize(score_pi)
    
    # Blend: primarily EI, with some UCB for exploration
    # Weight UCB more when sigma is relatively large
    sigma_mean = np.mean(sigma)
    if sigma_mean > eps:
        sigma_ratio = np.minimum(sigma / (sigma_mean + eps), 3.0) / 3.0
    else:
        sigma_ratio = np.zeros_like(sigma)
    
    # Dynamic weighting: more exploration when uncertainty is high
    w_ucb = 0.15 + 0.10 * sigma_ratio  # 0.15 to 0.25
    w_ei = 1.0 - w_ucb - 0.1  # remaining for EI
    w_pi = 0.1  # small weight for PI
    
    blended = w_ei * n_ei + w_ucb * n_ucb + w_pi * n_pi
    
    # Final score: use the blended score but scale by the actual EI magnitude
    # to preserve the relative improvement signal
    final_score = blended * (ei + 1e-6) / cost_safe
    
    # Ensure finite values
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    # Also ensure non-negative (higher is better)
    final_score = np.maximum(final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
