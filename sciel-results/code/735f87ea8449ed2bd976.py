import numpy as np
from scipy.special import erfc

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Pareto-filtered UCB with Expected Improvement.
    
    Strategy:
    1. Compute EI for exploitation signal.
    2. Compute UCB with adaptive beta for exploration.
    3. Apply Pareto filtering: mask out candidates whose predicted mean is
       significantly worse than f_best, restricting search to plausible improvement regions.
    4. Combine filtered UCB with EI for a balanced score.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Floor sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    sqrt_2 = np.sqrt(2.0)
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    
    # Phi(z) = 0.5 * erfc(-z/sqrt(2))
    Phi = 0.5 * erfc(-z_clipped / sqrt_2)
    # phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    phi = np.exp(-0.5 * z_clipped**2) / sqrt_2pi
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    EI = np.maximum(EI, 0.0)
    
    # Adaptive beta for UCB: higher when far from optimum (f_best large)
    # The Weierstrass function on [-0.5, 0.5] has values in roughly [0, 0.5]
    f_scale = 0.5
    f_norm = min(max(f_best, 0.0), f_scale) / f_scale  # in [0, 1]
    
    # Beta ranges from ~1.0 (exploitative) to ~3.5 (exploratory)
    beta = 1.0 + 2.5 * f_norm
    
    # UCB score for minimization: -mu + beta * sigma
    UCB = -mu + beta * sigma_safe
    
    # Pareto filtering: mask out candidates whose predicted mean is worse than
    # f_best by more than a tolerance. The tolerance scales with uncertainty.
    # Candidates with mu > f_best + tol are considered implausible for improvement.
    tol = 0.1 * f_norm + 0.01  # Small base tolerance, grows with exploration need
    pareto_mask = mu <= (f_best + tol)
    
    # Apply UCB only to filtered subset; set masked candidates to -inf
    UCB_filtered = np.where(pareto_mask, UCB, -np.inf)
    
    # Combine EI and filtered UCB
    # EI provides exploitation signal; UCB provides structured exploration
    # Weight UCB more heavily when f_best is large (early in optimization)
    w_ucb = 0.5 + 0.5 * f_norm
    
    # Final score: EI + weighted UCB (filtered)
    score = EI + w_ucb * UCB_filtered
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
