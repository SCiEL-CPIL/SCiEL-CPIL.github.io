"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on griewank 2d.
    Uses Expected Improvement per unit cost (EI / cost) as the core objective,
    which directly optimizes marginal return per unit of budget spent.
    """
    t0 = time.time()

    # Ensure inputs are valid
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Calculate z-score: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe

    # Probability of Improvement: P(f(x) < f_best)
    pi_prob = ndtr(z)

    # Expected Improvement: E[max(f_best - f(x), 0)]
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    ei = (f_best - mu) * pi_prob + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)

    # Core: EI per unit cost
    # This naturally balances improvement potential against evaluation cost
    ei_per_cost = ei / cost_safe

    # Add a small exploration term based on uncertainty, also cost-adjusted
    # This helps when EI is zero (e.g., f_best is very good) but there's still uncertainty
    # Use a normalized sigma to make it comparable
    sigma_norm = sigma_safe / (np.max(sigma_safe) + 1e-9)
    exploration = 0.01 * sigma_norm / cost_safe

    # Combine: primarily EI/cost, with a small exploration bonus
    score = ei_per_cost + exploration

    # Clamp non-finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard
    if time.time() - t0 > 280:
        return score

    return score
# EVOLVE-BLOCK-END
