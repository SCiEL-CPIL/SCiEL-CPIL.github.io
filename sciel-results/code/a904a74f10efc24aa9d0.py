"""Cost-aware Knowledge Gradient acquisition for Bayesian optimization on Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_KG_START_TIME = None
_KG_DEADLINE_S = 320.0  # Safety margin before 330s limit


def _kg_pair(mu1, s1, mu2, s2):
    """
    Compute E[max(X1, X2) - X1] where X1 ~ N(mu1, s1^2), X2 ~ N(mu2, s2^2).
    This is the expected reduction in the maximum (for minimization, 
    we want to minimize the best, so we look at min).
    
    For minimization: KG(x) = E[min(f_best, f(x))] - min(f_best, E[f(x)])
    But more precisely, if we have current best b, and evaluate x with
    f(x) ~ N(mu_x, sigma_x^2), the expected reduction in best is:
    E[min(b, f(x))] - min(b, mu_x) ... no.
    
    Actually, KG for minimization:
    Current best value is b (deterministic).
    New observation y ~ N(mu, sigma^2).
    New best value is min(b, y).
    Expected new best is E[min(b, y)].
    KG(x) = b - E[min(b, y)]
    
    E[min(b, y)] = integral_{-inf}^{b} y * p(y) dy + integral_{b}^{inf} b * p(y) dy
                 = integral_{-inf}^{b} y * p(y) dy + b * (1 - Phi((b - mu)/sigma))
    
    Let z = (b - mu) / sigma.
    integral_{-inf}^{b} y * p(y) dy = mu * Phi(z) - sigma * phi(z)
    where phi is the standard normal pdf and Phi is the standard normal cdf.
    
    So E[min(b, y)] = mu * Phi(z) - sigma * phi(z) + b * (1 - Phi(z))
                    = b - (b - mu) * Phi(z) - sigma * phi(z)
    
    KG(x) = b - [b - (b - mu) * Phi(z) - sigma * phi(z)]
          = (b - mu) * Phi(z) + sigma * phi(z)
    
    Note: b - mu can be positive or negative.
    If mu > b, z < 0, Phi(z) < 0.5.
    If mu < b, z > 0, Phi(z) > 0.5.
    """
    # Handle edge case where sigma is very small
    s = np.maximum(s1, 1e-9)
    z = (mu1 - mu2) / s  # Here mu1 is b, mu2 is mu_x, s is sigma_x
    # Wait, the formula above was for min(b, y) where y ~ N(mu2, s^2).
    # So mu1 in the function signature should be b (f_best), and mu2 should be mu_x.
    # Let's stick to the variable names in the function signature:
    # mu1: current best (b)
    # mu2: mean of new observation
    # s1: std of new observation (should be 0 for b, but we use it for scaling? No, b is deterministic)
    # Actually, the function signature _kg_pair(mu1, s1, mu2, s2) is a bit confusing.
    # Let's assume mu1 is the deterministic current best, and s1 is ignored or 0.
    # mu2 is the GP mean, s2 is the GP std.
    
    # To be safe, let's just use mu2 and s2.
    s = np.maximum(s2, 1e-9)
    z = (mu1 - mu2) / s
    
    # Avoid overflow/underflow in exp for large |z|
    # phi(z) = exp(-0.5 * z^2) / sqrt(2*pi)
    # For z > 50, phi(z) is effectively 0.
    # For z < -50, Phi(z) is effectively 0, but (mu1 - mu2) is large negative.
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    kg = (mu1 - mu2) * Phi_z + s * phi_z
    
    # Numerical stability: if z is very large positive, Phi_z ~ 1, kg ~ mu1 - mu2 + 0.
    # If z is very large negative, Phi_z ~ 0, kg ~ 0.
    # This seems correct.
    
    return kg


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Knowledge Gradient (KG) acquisition function.
    
    The score is the expected reduction in the best observed value,
    normalized by the evaluation cost.
    """
    global _KG_START_TIME
    
    if _KG_START_TIME is None:
        _KG_START_TIME = time.time()
        
    elapsed = time.time() - _KG_START_TIME
    if elapsed > _KG_DEADLINE_S:
        # Fallback to a simple heuristic if we're running out of time
        # Just return cost-aware LCB
        s = np.maximum(sigma, 1e-9)
        lcb = mu - 1.5 * s
        improvement = f_best - lcb
        c = np.maximum(cost, 1e-9)
        scores = improvement / c
        scores = np.where(np.isfinite(scores), scores, 0.0)
        scores = np.maximum(scores, 0.0)
        return scores

    # Ensure sigma is positive
    s = np.maximum(sigma, 1e-9)
    
    # Compute KG for each candidate
    # mu1 = f_best (scalar), mu2 = mu (array), s2 = s (array)
    kg = _kg_pair(np.full_like(mu, f_best), np.zeros_like(mu), mu, s)
    
    # Cost awareness: Divide by cost
    c = np.maximum(cost, 1e-9)
    scores = kg / c
    
    # Safety: Ensure finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
