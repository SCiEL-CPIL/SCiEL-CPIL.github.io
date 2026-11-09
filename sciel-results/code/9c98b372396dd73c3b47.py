"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural blend of Expected Improvement
    and Upper Confidence Bound (UCB), normalized by evaluation cost.
    
    Parameters
    ----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical stability
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    
    # Clip z to avoid extreme values
    z = np.clip(z, -10, 10)
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Compute UCB for minimization: we want low values, so UCB = mu - kappa * sigma
    # The improvement potential is f_best - (mu - kappa * sigma)
    # But we want a score that is high when improvement is high.
    # Let's use a "regret" style UCB: expected reduction in best value.
    # Actually, standard UCB for minimization chooses x to minimize mu - kappa*sigma.
    # To convert to a "score" (higher is better), we can use:
    # score_ucb = f_best - (mu - kappa * sigma)
    # This is positive when mu - kappa*sigma < f_best.
    
    # We need to choose kappa. A common heuristic is based on the number of evaluations,
    # but we don't have that directly. However, we can estimate it or use a fixed value.
    # Since cost is involved, let's use a fixed kappa that balances exploration.
    # The GP is on scaled coordinates [0,1]^2 with length scale 0.2.
    # A reasonable kappa might be around 1-3. Let's try 2.0.
    
    kappa = 2.0
    ucb = f_best - (mu - kappa * sigma)
    
    # Blend EI and UCB.
    # EI is more exploitation-focused, UCB is more exploration-focused.
    # We can normalize both to have similar scales.
    # Or simply weight them.
    
    # Let's try a weighted sum.
    # To make them comparable, we might divide by their respective scales.
    # But a simple weighted sum often works.
    
    # Let's use a blend: 0.5 * EI + 0.5 * UCB
    # But their scales might differ.
    # EI is roughly in the range of the function's variability.
    # UCB is also in the range of the function's variability.
    
    # Let's try: score = EI + UCB
    # Or perhaps: score = max(EI, UCB) ? No, that's not smooth.
    
    # Let's stick to a linear combination.
    # w_ei = 0.5, w_ucb = 0.5
    
    blended = 0.5 * ei + 0.5 * ucb
    
    # Cost-aware score: Maximize expected improvement per unit cost
    scores = blended / cost
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 295.0:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
