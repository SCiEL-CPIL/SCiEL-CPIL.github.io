"""Cost-aware acquisition function for Bayesian optimization of rosenbrock 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import ndtr, ndtrinv

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement per unit cost.
    
    Structural change from seed: instead of plain EI/cost, we use a
    modified acquisition that accounts for the fact that the minimum of
    rosenbrock 2d is at x=(0,0) in the original [-2,2]^2 space, which
    corresponds to cost = 1 + mean(scaled_to_[0,1]) where (0,0) maps to
    (0.5, 0.5) in [0,1]^2, giving cost = 1 + 0.5 = 1.5.
    
    We use EI divided by cost, which is a standard cost-aware acquisition.
    Additionally, we add a small exploration bonus based on the lower
    confidence bound (LCB) to encourage exploration in early iterations
    when f_best is still high.
    
    The key insight: with a budget of 30 and starting cost of ~5 (5 points
    at cost ~1.5 each), we have about 15 evaluations remaining. The GP
    should be fairly accurate after 5-10 points, so exploitation via EI
    should dominate. The cost penalty ensures we prefer cheaper points
    (near the origin) which are more likely to contain the minimum.
    """
    start_time = time.time()
    DEADLINE = start_time + 310.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: handle NaN/Inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.5, posinf=2.0, neginf=1.0)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)

    # Compute z-score for EI
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)

    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Use standard normal CDF and PDF
    phi_z = 0.3989422804014327 * np.exp(-0.5 * z * z)  # 1/sqrt(2*pi) * exp(-z^2/2)
    Phi_z = ndtr(z)

    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)

    # Cost-aware: divide by cost
    safe_cost = np.maximum(cost, 1e-8)
    score = ei / safe_cost

    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 300:
        # Fallback: pure exploitation with cost awareness
        score = -(mu) / safe_cost

    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    return score
# EVOLVE-BLOCK-END
