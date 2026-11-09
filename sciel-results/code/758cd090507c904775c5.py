"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: ensure we return within time limits
    start_time = time.time()
    
    # Strategy: Cost-Adjusted Expected Improvement (EI)
    #
    # Rationale:
    # 1. EI (Expected Improvement) is a standard, robust acquisition function that
    #    balances exploration (high sigma) and exploitation (low mu relative to f_best).
    #    It returns the expected value of max(f_best - f(x), 0).
    # 2. Unlike PI, which is a probability in [0,1], EI scales with the potential
    #    magnitude of improvement. This is crucial for Rosenbrock, where the
    #    "basin of attraction" is narrow and deep.
    # 3. Cost-awareness: The problem defines cost as 1 + mean(x_scaled).
    #    The minimum of Rosenbrock is at (1, 1) in standard coords, which maps to
    #    high cost. However, the path to the minimum often involves moving through
    #    lower cost regions initially.
    #    We use a cost penalty term. A common approach is EI / cost or EI - beta * cost.
    #    Given the scoring metric is simple regret, we want to minimize f(x).
    #    Dividing by cost encourages finding improvements in cheaper regions, which
    #    aligns with the fact that the valley starts in lower cost areas.
    #    However, if we are far from the optimum, high cost might be necessary.
    #    Let's use a soft penalty: score = EI / (1 + cost).
    #    This normalizes the improvement by the expense.
    
    # Compute EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # If sigma is very small, we handle it carefully to avoid division by zero.
    
    sigma_safe = np.clip(sigma, 1e-9, None)
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement
    # Note: If mu > f_best, z is negative. Phi(z) is small, phi(z) is small.
    # The formula holds generally.
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware adjustment
    # We want to maximize EI per unit cost.
    # cost is in range [1, 2] roughly (since x is in [0,1], mean is in [0,1], cost = 1+mean).
    # So dividing by cost is a mild penalty.
    # Let's try score = EI / cost
    # Or perhaps a more aggressive penalty if cost is high?
    # The current PI used pi / (1+cost).
    # Let's stick to EI / cost for a direct "value per cost" metric.
    
    score = ei / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - start_time > 320:
        # Fallback: return the current scores. The computation above is vectorized
        # and should be very fast (< 1ms for 2000 points), so this is unlikely.
        pass

    return score
# EVOLVE-BLOCK-END
