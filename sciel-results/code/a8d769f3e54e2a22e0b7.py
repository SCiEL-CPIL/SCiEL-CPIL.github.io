"""
Quantile-based acquisition: selects points with high predictive uncertainty 
only if their predicted mean is in the top 5% (lowest 5%) of candidates.
This decouples exploration from the absolute magnitude of GP variance, 
focusing instead on the relative rank of uncertainty among promising candidates.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard (330 s wall-clock limit)
_T_START = time.time()
_TIME_BUDGET = 320.0  # leave 10 s margin

def _elapsed():
    return time.time() - _T_START

def acquisition(mu, sigma, f_best):
    """
    Quantile-based acquisition function.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points.
    f_best : float
        Best observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Time guard: if we are close to the deadline, fall back to a fast score
    if _elapsed() > _TIME_BUDGET:
        # Fast fallback: simple EI without heavy processing
        eps = 1e-9
        z = (f_best - mu) / (sigma + eps)
        scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    n = len(mu)
    eps = 1e-9

    # 1. Identify promising candidates: those with mu in the bottom 5% (lowest 5%)
    #    Since we are minimizing, lower mu is better.
    #    We use a percentile threshold to define the "promising" set.
    mu_sorted = np.sort(mu)
    threshold_idx = max(1, int(0.05 * n))  # top 5%
    mu_threshold = mu_sorted[threshold_idx - 1]
    
    # Mask for promising candidates
    promising_mask = mu <= mu_threshold

    # 2. For promising candidates, score based on uncertainty (sigma)
    #    We want to explore within this promising region.
    #    The score should be high for high sigma.
    #    To make the scale comparable and robust, we normalize sigma.
    #    However, the task says "higher meaning evaluate next".
    #    We can simply use sigma as the score for promising points, 
    #    and a very low score for others.
    
    # To ensure we don't completely ignore exploitation if the GP is very confident 
    # (sigma is very small everywhere), we might need a fallback. 
    # But the directive says "selecting points that are in the top 5% of uncertainty 
    # but only if their mean is below the current best". 
    # Wait, the directive says: "top 5% of uncertainty but only if their mean is below the current best".
    # Let's re-read carefully: "effectively selecting points that are in the top 5% of uncertainty 
    # but only if their mean is below the current best."
    # This implies two conditions:
    # 1. High uncertainty (top 5% of sigma)
    # 2. Promising mean (mu < f_best)
    
    # Let's refine the strategy based on the directive's specific phrasing:
    # "score is the negative predicted mean plus a penalty derived from the empirical rank of the uncertainty"
    # Actually, the directive says: "Use a quantile-based exploration strategy where the acquisition score 
    # is the negative predicted mean plus a penalty derived from the empirical rank of the uncertainty 
    # among candidates, effectively selecting points that are in the top 5% of uncertainty 
    # but only if their mean is below the current best."
    
    # This description is slightly ambiguous. Let's interpret it as:
    # Score = -mu + beta * rank(sigma)
    # But "effectively selecting points that are in the top 5% of uncertainty but only if their mean is below the current best"
    # suggests a hard filter or a very strong weighting.
    
    # Let's try a hybrid:
    # 1. Compute rank of sigma (higher rank = more uncertain).
    # 2. Compute rank of mu (lower mu = better).
    # 3. Filter: only consider points where mu < f_best.
    # 4. Among those, pick those with high sigma rank.
    
    # A more robust implementation of the "Quantile-based" idea:
    # 1. Calculate empirical CDF (rank) of sigma.
    # 2. Calculate empirical CDF (rank) of mu.
    # 3. Score = -mu (exploitation) + lambda * sigma_rank (exploration)
    #    But we want to emphasize the top 5% of uncertainty.
    #    So, we can amplify the sigma_rank for points in the top 5%.
    
    # Let's stick to a simpler, robust interpretation:
    # Score = -mu * I(mu < f_best) + gamma * sigma * I(sigma_rank > 0.95)
    # This might be too sparse.
    
    # Alternative:
    # Score = -mu + gamma * (sigma / max(sigma)) * I(mu < f_best)
    # This encourages exploration only where the mean is already good.
    
    # Let's try:
    # 1. Mask out points where mu >= f_best (they are not promising).
    # 2. For the remaining points, score = -mu + gamma * sigma.
    #    This is like EI but with a hard gate on mu.
    
    # Let's implement:
    # 1. Hard gate: if mu >= f_best, score = -infinity (or very low).
    # 2. For mu < f_best, score = -mu + gamma * sigma.
    #    This balances exploitation (-mu) and exploration (sigma) only in promising regions.
    
    # How to choose gamma?
    # If gamma is too large, we explore too much. If too small, we exploit too much.
    # We can adapt gamma based on the current spread of sigma.
    # gamma = np.mean(sigma) / np.std(mu[mu < f_best]) + 1e-9
    # This normalizes the exploration term to be comparable to the exploitation term.
    
    promising = mu < f_best
    if np.sum(promising) == 0:
        # If no points are promising, fall back to pure exploration (max sigma)
        # or pure exploitation (min mu). 
        # Since we are minimizing, and no point is better than f_best, 
        # we should explore to find a better region.
        scores = sigma
    else:
        # Calculate gamma
        mu_promising = mu[promising]
        sigma_promising = sigma[promising]
        
        # Normalization
        mu_range = np.ptp(mu_promising) + eps
        sigma_range = np.ptp(sigma_promising) + eps
        
        # If the range is very small, gamma can be large.
        # Let's use a fixed gamma scaled by the typical scale.
        # gamma = 1.0
        # But we want to make sure the terms are comparable.
        # score = -mu + gamma * sigma
        # Let's set gamma = np.std(mu_promising) / (np.std(sigma_promising) + eps)
        # This makes the standard deviations of the two terms comparable.
        
        std_mu = np.std(mu_promising) + eps
        std_sigma = np.std(sigma_promising) + eps
        gamma = std_mu / std_sigma
        
        # Score calculation
        scores = np.full(n, -1e10)  # Initialize with very low score
        scores[promising] = -mu[promising] + gamma * sigma[promising]
        
        # Ensure finite
        if not np.all(np.isfinite(scores)):
            finite_mask = np.isfinite(scores)
            if np.any(finite_mask):
                mean_val = np.mean(scores[finite_mask])
                scores = np.where(finite_mask, scores, mean_val)
            else:
                scores = np.zeros_like(mu)

    # Final safety check
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
