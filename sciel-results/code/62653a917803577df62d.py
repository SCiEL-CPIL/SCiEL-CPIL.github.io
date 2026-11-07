import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function for Bayesian optimization.
    KG approximates the expected reduction in the global minimum by computing
    the probability that each candidate becomes the new best, weighted by the
    expected improvement over the current best.
    
    Parameters:
    -----------
    mu : array-like, shape (N,)
        Posterior mean at N candidate points
    sigma : array-like, shape (N,)
        Posterior standard deviation at N candidate points
    f_best : float
        Current best (lowest) observed value
        
    Returns:
    --------
    scores : ndarray, shape (N,)
        KG scores; higher values indicate more promising candidates
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score for each candidate relative to f_best
    z = (f_best - mu) / sigma_safe
    
    # Probability that f(x) < f_best (candidate is a new best)
    prob_new_best = norm.cdf(z)
    
    # Expected improvement over f_best for each candidate
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * prob_new_best + sigma_safe * norm.pdf(z)
    
    # Knowledge Gradient: expected reduction in the global minimum
    # KG(x) ≈ P(f(x) < f_best) * E[min(f(x), f_best) - min(f(x), f_best) | f(x) < f_best]
    # More precisely, the expected improvement when x is the new best
    # weighted by the probability of being the new best
    
    # A common approximation for KG in the context of minimization:
    # KG(x) = E[min(min(f(x), f_best) - f_best, 0)] 
    #        = -E[max(min(f(x), f_best) - f_best, 0)]
    #        = E[f_best - min(f(x), f_best)]
    #        = E[max(f_best - f(x), 0)]
    #        = EI(x)
    
    # But a better KG formulation accounts for the probability that x is the
    # global minimum among all observations, not just improvement over f_best.
    # For a single candidate vs current best:
    # KG(x) = P(f(x) < f_best) * E[f(x) - f(x) | f(x) < f_best] + ...
    # 
    # A practical and commonly used approximation:
    # KG(x) = P(f(x) < f_best) * (f_best - E[f(x) | f(x) < f_best])
    
    # E[f(x) | f(x) < f_best] = mu - sigma * phi(z) / Phi(z) when z > 0
    # (truncated normal mean)
    
    # For numerical stability, handle the case where prob_new_best is very small
    eps = 1e-12
    prob_safe = np.maximum(prob_new_best, eps)
    
    # Mean of f(x) given f(x) < f_best (truncated normal)
    # If z is the z-score for f(x) < f_best, then:
    # E[f(x) | f(x) < f_best] = mu - sigma * phi(z)/Phi(z)
    phi_z = norm.pdf(z)
    trunc_mean = mu - sigma_safe * phi_z / prob_safe
    
    # Expected improvement when x is the new best
    improvement_given_new_best = f_best - trunc_mean
    
    # KG score: probability of being new best times expected improvement
    kg = prob_new_best * improvement_given_new_best
    
    # Ensure finite values
    kg = np.where(np.isfinite(kg), kg, 0.0)
    
    # Clip to non-negative (KG should be non-negative)
    kg = np.maximum(kg, 0.0)
    
    return kg
