import numpy as np
import time
from scipy.special import erf

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    DEADLINE = 320.0
    
    n = mu.shape[0]
    
    # Clamp sigma for numerical stability
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Clamp cost for numerical stability
    cost_safe = np.maximum(cost, 1e-10)
    
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = exp(-0.5 * z^2) / sqrt(2 * pi)
    
    # Use np.clip to avoid overflow in exp for very negative z, though z can be large positive or negative.
    # If z is very negative, Phi(z) ~ 0, phi(z) ~ 0.
    # If z is very positive, Phi(z) ~ 1, phi(z) ~ 0.
    
    # Vectorized calculation
    # To prevent overflow in exp, we can clip z.
    # exp(-0.5 * z^2) underflows to 0 for z > ~ 40, which is fine.
    # It overflows? No, exp of negative is safe.
    
    # However, if z is very large positive, z^2 is large positive, -0.5*z^2 is large negative -> exp is 0.
    # If z is very large negative, z^2 is large positive, -0.5*z^2 is large negative -> exp is 0.
    
    # Calculate Phi and phi
    Phi_z = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    phi_z = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    EI = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Cost-aware score
    # Higher EI / cost is better
    scores = EI / cost_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Fallback: simple cost-aware LCB
        beta = 1.0
        ucb = mu - beta * sigma_safe
        simple = -ucb / cost_safe
        return np.where(np.isfinite(simple), simple, 0.0)
        
    return scores
