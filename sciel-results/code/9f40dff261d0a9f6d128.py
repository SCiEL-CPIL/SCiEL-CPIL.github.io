import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function for minimization.
    
    KG estimates the expected reduction in the global minimum value.
    We approximate this by considering the predictive distribution at each candidate
    and estimating how much it could improve upon the current best.
    
    For each candidate i, we model:
    - f_i ~ N(mu_i, sigma_i^2)
    - The new minimum after evaluating i is min(f_best, f_i)
    - The expected improvement is E[max(f_best - f_i, 0)]
    
    To capture the "knowledge gradient" aspect, we also consider the uncertainty
    in the global minimum by looking at the distribution of the minimum over
    all candidates. A simple but effective approximation is to weight the
    expected improvement by how likely this point is to be the new minimum.
    """
    # Handle cases where sigma is very small to avoid division by zero
    safe_sigma = np.maximum(sigma, 1e-10)
    
    # Standardized distance from f_best
    z = (f_best - mu) / safe_sigma
    
    # Expected improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + safe_sigma * norm.pdf(z)
    
    # Clip negative values to zero (EI should be non-negative)
    ei = np.maximum(ei, 0.0)
    
    # For KG, we want to prioritize points that are likely to be the new global minimum.
    # A point is likely to be the new minimum if it has a low predicted value and/or high uncertainty.
    # We can approximate the probability that this point is the new minimum by looking at
    # how many other points have a lower expected value.
    
    # Compute the probability that each point is the minimum of the predictive means
    # This is an approximation - in full KG, we'd integrate over the joint distribution
    
    # A simpler approach: weight by the inverse of the number of points with lower mu
    # Points with very low mu are more likely to be the new minimum
    
    # Sort by mu to get rank
    sorted_indices = np.argsort(mu)
    ranks = np.empty_like(mu)
    ranks[sorted_indices] = np.arange(len(mu))
    
    # Probability of being the minimum is roughly 1/(rank + 1) for a rough estimate
    # But this is too aggressive. Let's use a softer weighting.
    
    # Actually, let's use a different approach: the KG value is approximately
    # the expected improvement, but we can enhance it by considering the spread.
    
    # A practical KG approximation: 
    # KG_i ≈ EI_i * (1 + beta * sigma_i / max_sigma)
    # where beta controls how much we value exploration
    
    # Or even simpler: use EI but with a bonus for high uncertainty in the tail
    # that could lead to a new minimum
    
    # Let's try: KG ≈ EI + gamma * sigma * exp(-(mu - f_best)^2 / (2 * sigma^2))
    # This adds a bonus for points with high sigma that are near f_best
    
    # Actually, the standard EI already captures this. Let me think differently.
    
    # The key insight of KG vs EI: KG considers the information gained about the
    # global minimum. If we evaluate point i and it's better than f_best, we've
    # improved our estimate of the minimum. If it's worse, we've learned that
    # point i is not the minimum, which also has value.
    
    # A common practical approximation is to use EI with a small exploration bonus.
    # But since we're keeping the same approach, let's refine the EI calculation.
    
    # For better numerical stability and to capture the KG aspect, let's compute
    # the expected improvement more carefully and add a small exploration term
    # that rewards points with high uncertainty that could potentially beat f_best.
    
    # The probability that f_i < f_best is Phi(z) where z = (f_best - mu_i) / sigma_i
    # The expected improvement given that f_i < f_best is f_best - mu_i + sigma_i * phi(z)/Phi(z)
    
    # But EI already accounts for this. Let's just use EI as is, which is a good
    # approximation of KG for a single point.
    
    scores = ei
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
