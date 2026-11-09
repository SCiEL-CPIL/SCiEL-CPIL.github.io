import numpy as np
import time

# Deadline guard: return early if we're running too long
_START_TIME = time.time()
_DEADLINE = 330.0  # Wall-clock limit is 330s, but each call should be fast
_MARGIN = 5.0      # Safety margin

def _time_check():
    """Check if we're close to the deadline."""
    return (time.time() - _START_TIME) > (_DEADLINE - _MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    mu, sigma: shape (2000,), posterior mean and std in objective units.
    f_best: lowest observed value.
    
    Returns 2000 finite scores, higher means evaluate next.
    """
    # Check deadline
    if _time_check():
        # Return a simple fallback: UCB with beta=0.5
        return 0.5 * sigma - mu
    
    n = len(mu)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is standard normal CDF and phi is standard normal PDF
    
    # But we can also use a hybrid approach:
    # - When sigma is large, explore (UCB-like)
    # - When sigma is small, exploit (EI-like)
    
    # Let's use a weighted combination that adapts based on the current state
    
    # First, compute EI
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    # Use scipy for accuracy
    from scipy.stats import norm
    EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB with adaptive beta
    # Use a smooth, data-driven beta that balances exploration/exploitation
    # 
    # Key insight: The optimal beta should reflect the uncertainty in f_best.
    # As we collect more data, f_best becomes more reliable, so we can reduce
    # exploration. We estimate "reliability" from the spread of mu values.
    #
    # Strategy:
    # 1. Compute a normalized uncertainty measure from sigma distribution
    # 2. Scale beta based on how much improvement is still possible (f_best vs 0)
    # 3. Blend EI and UCB smoothly based on local uncertainty
    
    # Measure overall uncertainty level
    sigma_median = np.median(sigma)
    sigma_max = np.max(sigma)
    sigma_min = np.min(sigma)
    sigma_range = sigma_max - sigma_min + 1e-12
    
    # Normalize sigma to [0, 1] for local weighting
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Base beta: start with moderate exploration, decay as f_best approaches 0
    # Ackley 2D values are roughly in [0, ~20], so normalize f_best by ~5
    f_best_scale = max(f_best, 1e-6)
    
    # Beta decreases smoothly as f_best gets smaller
    # When f_best > 5, beta ~ 2.0 (heavy exploration)
    # When f_best ~ 1, beta ~ 1.0 (balanced)
    # When f_best < 0.1, beta ~ 0.3 (heavy exploitation)
    beta = 0.3 + 1.7 * np.exp(-f_best / 2.0)
    
    UCB = beta * sigma - mu
    
    # Smooth blend: weight UCB more where sigma is high, EI where sigma is low
    # Use a sigmoid-like transition based on normalized sigma
    # Points with high sigma get more UCB weight, low sigma get more EI weight
    blend_temp = 0.5  # Controls transition sharpness
    w_ucb = 1.0 / (1.0 + np.exp(-blend_temp * (sigma_norm - 0.5)))
    
    scores = w_ucb * UCB + (1 - w_ucb) * EI
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
