"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a hybrid of Expected Improvement and Probability of Improvement,
    with adaptive weighting based on search progress and cost normalization.
    
    Args:
        mu: Array of shape (N,) with GP mean predictions.
        sigma: Array of shape (N,) with GP standard deviations.
        f_best: Scalar, the best observed function value so far.
        cost: Array of shape (N,) with evaluation costs for each candidate.
        
    Returns:
        Array of shape (N,) with scores. Higher score is better.
    """
    start_time = time.time()
    deadline = start_time + 280.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    f_best_val = float(f_best)

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-8)

    # Compute z-scores: how many standard deviations below f_best the mean is
    z = (f_best_val - mu) / sigma_safe
    # Clip z for numerical stability in CDF/PDF evaluation
    z = np.clip(z, -15.0, 15.0)
    
    # Expected Improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best_val - mu) * Phi_z + sigma_safe * phi_z
    # EI should be non-negative; clip tiny negative values from numerical noise
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PI):
    # PI = Phi(z) = P(f(x) < f_best)
    pi = Phi_z
    
    # Lower Confidence Bound (LCB) for additional exploitation signal
    # Use a moderate, fixed beta for stability
    beta = 1.0
    lcb = mu - beta * sigma_safe
    
    # Cost-adjusted component scores
    ei_score = ei / cost_safe
    pi_score = pi / cost_safe
    lcb_score = -lcb / cost_safe  # Negate because lower LCB is better
    
    # Adaptive combination: 
    # Early on (high f_best), explore more via PI and LCB
    # Later (low f_best), exploit more via EI
    f_best_clipped = np.clip(f_best_val, 0.0, 1.0)
    
    # Weight EI heavily when close to optimum
    w_ei = 0.6 + 0.4 * (1.0 - f_best_clipped)
    # Weight PI for balanced exploration/exploitation
    w_pi = 0.2 + 0.2 * f_best_clipped
    # Weight LCB for uncertainty-driven exploration
    w_lcb = 0.2
    
    # Normalize weights
    w_sum = w_ei + w_pi + w_lcb
    w_ei /= w_sum
    w_pi /= w_sum
    w_lcb /= w_sum
    
    scores = w_ei * ei_score + w_pi * pi_score + w_lcb * lcb_score
    
    # Deadline guard: if we're running too long, fall back to simple exploitation
    if time.time() > deadline:
        scores = -mu / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
