import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, improvement is f_best - f(x).
    # If f(x) < f_best, improvement is positive.
    # The standard EI formula for maximization is (mu - f_best)Phi(z) + sigma phi(z) with z=(mu-f_best)/sigma.
    # For minimization, we flip the sign of the improvement term.
    # Let's derive carefully.
    # We want E[max(0, f_best - f(x))]
    # Let Y = f(x) ~ N(mu, sigma^2).
    # E[max(0, f_best - Y)]
    # Let Z = (Y - mu)/sigma. Y = mu + sigma Z.
    # max(0, f_best - mu - sigma Z)
    # If f_best - mu > 0 (let D = f_best - mu > 0):
    #   max(0, D - sigma Z)
    #   Integral from -inf to D/sigma of (D - sigma z) phi(z) dz
    #   = D * Phi(D/sigma) - sigma * phi(D/sigma)
    # If f_best - mu < 0 (let D = f_best - mu < 0):
    #   D - sigma Z is always negative? No.
    #   If D < 0, D - sigma Z < 0 for Z > D/sigma (which is negative).
    #   Wait, if D < 0, the term inside max is D - sigma Z.
    #   It is positive when D - sigma Z > 0 => Z < D/sigma.
    #   Since D < 0, D/sigma < 0.
    #   Integral from -inf to D/sigma of (D - sigma z) phi(z) dz
    #   = D * Phi(D/sigma) - sigma * phi(D/sigma)
    #   Wait, the formula is the same?
    #   Let's check standard EI for minimization.
    #   Many sources define EI_min = (f_best - mu)Phi(z) + sigma phi(z) where z = (f_best - mu)/sigma.
    #   Let's verify limits.
    #   If mu << f_best (good), z is large positive. Phi(z)~1, phi(z)~0. EI ~ f_best - mu. Correct.
    #   If mu >> f_best (bad), z is large negative. Phi(z)~0, phi(z)~0. EI ~ 0. Correct.
    #   So the formula EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z) is correct for minimization.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    
    # Use square root of cost to reduce sensitivity to evaluation cost
    scores = ei / np.sqrt(cost_safe)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
