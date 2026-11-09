"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import ndtr

_START_TIME = time.time()
_DEADLINE = 330 - 5  # Safety margin: kill program 5s before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Strategy: Use the deterministic Expected Improvement, scaled by the cost of evaluation.
    EI is well-suited for continuous optimization and avoids the high variance of Thompson Sampling.
    The cost-aware adjustment (EI / cost) ensures that we prioritize points that offer the best
    improvement per unit of cost, which is crucial given the variable cost structure.
    
    We also add a small exploration bonus based on sigma to prevent the algorithm from getting
    stuck in local minima, especially in the early stages when the GP is uncertain.
    """
    # Deadline guard: if we are running out of time, return a simple greedy score
    if time.time() > _DEADLINE:
        # Fallback: prioritize low mu and low cost to finish quickly
        # This is a deterministic greedy approach
        return -mu / (cost + 1e-9)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu, sigma^2)
    
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Use scipy.special.ndtr for Phi
    # phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    
    # Clip z to avoid numerical issues in exp
    z_clipped = np.clip(z, -10, 10)
    
    phi_z = np.exp(-0.5 * z_clipped**2) / np.sqrt(2 * np.pi)
    Phi_z = ndtr(z_clipped)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0)
    
    # Cost-aware score: EI / cost
    # This encourages finding improvements that are "cheap" to evaluate
    cost_score = ei / cost_safe
    
    # Add a small exploration bonus to encourage visiting uncertain regions
    # This helps escape local minima, especially in multimodal functions
    # The bonus is proportional to sigma and inversely proportional to cost
    # We use a small weight (0.1) to keep exploitation dominant
    exploration_bonus = 0.1 * sigma_safe / cost_safe
    
    scores = cost_score + exploration_bonus
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
# EVOLVE-BLOCK-END
