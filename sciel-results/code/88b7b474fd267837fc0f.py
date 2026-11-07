"""Acquisition function using a structural shift to Thompson Sampling with local penalty."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0  # 10s safety margin before 330s kill

def _check_deadline():
    return time.time() - _start_time > _DEADLINE

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores using a Thompson Sampling-like approach
    with a local repulsion penalty to encourage exploration in multi-modal landscapes.
    
    Args:
        mu: Array of mean predictions, shape (2000,)
        sigma: Array of standard deviation predictions, shape (2000,)
        f_best: Current best observed function value (minimum)
        
    Returns:
        Array of scores, shape (2000,). Higher scores are preferred.
    """
    if _check_deadline():
        # Fallback: return EI scores quickly
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        from scipy.stats import norm
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: Sample from the posterior
    # We want to minimize, so we sample a value and pick the lowest
    # But we need to return scores for all 2000 points
    # Score = -sampled_value (higher is better)
    
    # Generate random samples
    # Use a single random draw for consistency within this call
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(mu.shape)
    
    # The acquisition score is the negative of the sampled value
    # This naturally balances exploration (high sigma) and exploitation (low mu)
    ts_scores = -samples
    
    # Add a local penalty to avoid re-evaluating near the current best
    # We don't have the coordinates, but we can approximate by penalizing points
    # where the GP is very confident and close to f_best
    # Penalty: if mu is close to f_best and sigma is low, reduce score
    
    # Distance in objective space from f_best
    # We want to penalize points that are "too similar" to the current best
    # in terms of predicted performance
    dist_to_best = np.abs(mu - f_best)
    
    # Points that are very close to f_best in prediction AND have low uncertainty
    # are likely already explored
    # Create a penalty that is high when dist_to_best is small and sigma is small
    # penalty = exp(-dist_to_best / sigma) * sigma
    # This is a soft penalty
    
    # Normalize dist_to_best by sigma to get a z-score-like quantity
    z_dist = dist_to_best / sigma_safe
    
    # Penalty: points with small z_dist (close to best) get penalized
    # But we want to keep some exploration, so use a mild penalty
    # penalty = np.exp(-z_dist**2 / 2) * (sigma_safe / np.max(sigma_safe + 1e-8))
    
    # Actually, a simpler approach: use the probability that the sample is better than f_best
    # P(sample < f_best) = Phi((f_best - mu) / sigma)
    from scipy.stats import norm
    prob_better = norm.cdf((f_best - mu) / sigma_safe)
    
    # Combine: TS score minus a penalty for high probability of being already explored
    # The penalty should be stronger for points that are very likely to be near f_best
    # and have low uncertainty
    
    # Use a penalty proportional to prob_better * (1 - normalized_sigma)
    sigma_norm = sigma_safe / (np.max(sigma_safe) + 1e-8)
    penalty = prob_better * (1 - sigma_norm) * np.max(ts_scores)
    
    scores = ts_scores - penalty
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
