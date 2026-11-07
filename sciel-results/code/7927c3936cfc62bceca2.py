"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track global start time for deadline safety
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to spare before 330s kill

# Constants
MIN_POSSIBLE = -0.8
MAX_POSSIBLE = 1.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Hybrid cEI + Adaptive Exploration.
    
    Structural Change:
    Combines Cost-Aware Expected Improvement (cEI) with an adaptive exploration term.
    The exploration term's weight is scaled by the "gap" between the current best
    observation and the known theoretical minimum. This ensures aggressive exploration
    (rewarding high uncertainty) in the early stages of optimization when the GP model
    is less reliable and the best observed value is far from the optimum, and a shift
    to pure exploitation (cEI) in the later stages when the model is more accurate and
    we are close to the minimum.
    
    This addresses the potential mode-collapse of pure EI in high dimensions by
    explicitly injecting an exploration signal that decays as progress is made.
    """
    # Deadline guard
    if time.time() - _START_TIME > 330.0 - _DEADLINE_MARGIN:
        return np.zeros_like(mu)

    n = len(mu)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1.0)
    
    # 1. Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -20, 20)
    
    # Phi(z) and phi(z)
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # EI calculation
    ei = diff * Phi + sigma_safe * phi
    
    # Clamp EI to be non-negative (standard EI definition)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    c_ei = ei / cost_safe
    
    # 2. Calculate Adaptive Exploration Term
    # The "gap" represents how far we are from the theoretical minimum.
    # High gap -> Early stage -> More exploration needed.
    # Low gap -> Late stage -> Exploit.
    
    gap = f_best - MIN_POSSIBLE
    total_range = MAX_POSSIBLE - MIN_POSSIBLE
    
    if total_range <= 0:
        exploration_scale = 0.0
    else:
        # Normalize gap to [0, 1]
        # If f_best is at MAX, gap is MAX-MIN, scale is 1.
        # If f_best is at MIN, gap is 0, scale is 0.
        exploration_scale = np.clip(gap / total_range, 0.0, 1.0)
    
    # Exploration coefficient: 
    # We want this to be significant enough to influence the choice early on.
    # sigma is in objective units. 
    # A value of 0.5 * sigma is a reasonable exploration bonus (approx 0.5 sigma move).
    # We divide by cost to maintain cost-efficiency.
    exploration_coeff = 0.5
    
    exploration_term = exploration_scale * exploration_coeff * (sigma_safe / cost_safe)
    
    # 3. Combine
    # The score is the sum of the exploitation signal (cEI) and the exploration signal.
    score = c_ei + exploration_term
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
