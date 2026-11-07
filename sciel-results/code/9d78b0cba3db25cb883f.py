import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Soft-min weighted Knowledge Gradient acquisition function.
    
    Structural change: weights the per-candidate KG by the probability
    that the candidate is the argmin among all 2000 candidates, using
    a softmax over -mu/sigma. This accounts for competition among
    candidates and focuses evaluation on points likely to set a new record.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation at candidate points
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Per-candidate Knowledge Gradient
    d = (f_best - mu) / sigma_safe
    phi_d = norm.pdf(d)
    Phi_d = norm.cdf(d)
    kg = (f_best - mu) * Phi_d + sigma_safe * phi_d
    
    # Structural: soft-min probability weighting
    # P(candidate i is argmin) ~ softmax(-mu_i / sigma_i)
    # Lower mu and higher sigma increase the probability
    logits = -mu / sigma_safe
    logits_max = np.max(logits)
    exp_logits = np.exp(logits - logits_max)
    
    # Weight KG by soft-min probability (unnormalized is fine for ranking)
    scores = kg * exp_logits
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
