import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a combination of Expected Improvement and cost-awareness.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    cost = np.maximum(cost, 1e-8)
    
    # Normalize sigma for stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Z-score: how far below f_best the mean is, in units of sigma
    # For minimization, we want mu < f_best
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    # Use scipy if available, otherwise approximate
    try:
        from scipy.stats import norm
        phi = norm.pdf(z)
        Phi = norm.cdf(z)
    except ImportError:
        # Approximate using numpy
        # phi(z) = exp(-0.5*z^2) / sqrt(2*pi)
        phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        # Phi(z) approximation using erf
        Phi = 0.5 * (1 + np.erf(z / np.sqrt(2)))
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: (f_best - mu) = z * sigma
    # So EI = z * sigma * Phi(z) + sigma * phi(z) = sigma * (z * Phi(z) + phi(z))
    ei = sigma_safe * (z * Phi + phi)
    
    # Cost-aware scaling
    # We want to penalize high cost points
    # A common approach is to divide by cost or use a power of cost
    # Since cost is in [1, 2] (1 + mean(x scaled to [0,1])), 
    # dividing by cost gives a reasonable penalty
    
    # Adaptive beta for exploration
    # Early on (high f_best), we want more exploration
    # Later (low f_best), we want more exploitation
    
    # f_best is in objective units. Ackley 2D range is [0, ~29]
    # Normalize f_best to [0, 1] for adaptive beta
    f_scale = 25.0
    f_ratio = np.clip(f_best / f_scale, 0.0, 1.0)
    
    # Beta: higher when f_best is high (early), lower when f_best is low (late)
    beta_min = 1.0
    beta_max = 3.0
    beta = beta_min + (beta_max - beta_min) * f_ratio
    
    # Combine EI with a UCB-like term
    # Score = EI / cost + beta * sigma / cost
    # This balances exploitation (EI) and exploration (sigma)
    
    # Actually, let's use a simpler and more robust approach:
    # Score = (EI + beta * sigma) / cost
    # This is like a cost-aware UCB with an EI component
    
    # Or even simpler: just use cost-aware EI with a small exploration bonus
    score = (ei + 0.5 * sigma_safe) / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
