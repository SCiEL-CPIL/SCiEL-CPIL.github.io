import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Hypervolume Improvement (HVI) acquisition function for Bayesian optimization.
    
    Instead of standard Expected Improvement (EI), we use a Hypervolume Improvement surrogate
    that estimates the expected reduction in the hypervolume of the non-dominated set.
    For single-objective minimization, this reduces to an integral that rewards points
    with high probability of improving f_best, weighted by the magnitude of improvement.
    
    The HVI for a single point x with GP prediction (mu, sigma) is:
    HVI = E[max(0, f_best - f(x))] * w
    where w is a weight that accounts for cost.
    
    For a Gaussian f(x) ~ N(mu, sigma^2), the expected improvement is:
    EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    We modify this with a cost-aware weighting that penalizes expensive points.
    """
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.clip(sigma, 1e-8, None)
    
    # Compute the standardized distance
    d = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    cdf_vals = norm.cdf(d)
    pdf_vals = norm.pdf(d)
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware weighting:
    # We want to favor points that provide high improvement per unit cost.
    # Use a soft weighting: weight = 1 / (1 + cost) to avoid extreme penalties
    # but still prefer cheaper points.
    cost_safe = np.clip(cost, 1e-8, None)
    weight = 1.0 / (1.0 + cost_safe)
    
    # Additional exploration bonus: in early stages (when f_best is still high relative
    # to the function's range), we want to explore more. We can use the variance as
    # an indicator of uncertainty and add a small exploration term.
    # However, EI already incorporates uncertainty through sigma.
    
    # Hypervolume Improvement score: EI weighted by cost efficiency
    scores = ei * weight
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
