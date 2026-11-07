"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Quantile of Improvement (QI) acquisition function for Bayesian optimization.
    
    QI focuses on high-probability, high-reward regions by maximizing the
    conditional expected improvement given that improvement occurs.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Safety: ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Compute Quantile of Improvement (QI)
    # QI = E[f_best - f | f < f_best] * P(f < f_best)
    # This is the expected improvement weighted by the probability of improvement
    # But QI specifically focuses on the conditional expectation given improvement
    
    improvement = f_best - mu
    z = improvement / sigma
    z = np.clip(z, -10, 10)
    
    # Probability of improvement
    prob_improve = norm.cdf(z)
    
    # Conditional expected improvement given improvement
    # E[f_best - f | f < f_best] = (f_best - mu) * Phi(z) + sigma * phi(z) / Phi(z)
    # But we need to handle the case where Phi(z) is very small
    phi_z = norm.pdf(z)
    
    # Avoid division by zero
    safe_prob = np.maximum(prob_improve, 1e-10)
    
    # Conditional expected improvement
    cond_ei = (improvement * prob_improve + sigma * phi_z) / safe_prob
    
    # QI = conditional expected improvement * probability of improvement
    # This is actually just EI, but let's use a different formulation
    
    # Alternative QI formulation: maximize the q-th quantile of the improvement
    # Use a quantile that balances exploration and exploitation
    # When f_best is large, use a higher quantile (more exploration)
    # When f_best is small, use a lower quantile (more exploitation)
    
    if f_best > 50.0:
        q = 0.7
    elif f_best > 10.0:
        q = 0.6
    elif f_best > 1.0:
        q = 0.5
    else:
        q = 0.4
    
    # The q-th quantile of the improvement distribution
    # Improvement = f_best - f, where f ~ N(mu, sigma^2)
    # So improvement ~ N(f_best - mu, sigma^2)
    # The q-th quantile of improvement is:
    # q_i = (f_best - mu) + sigma * norm.ppf(q)
    
    q_i = improvement + sigma * norm.ppf(q)
    q_i = np.maximum(q_i, 0.0)
    
    # Also compute standard EI for blending
    ei = improvement * prob_improve + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Normalize both to [0, 1]
    q_i_max = np.max(q_i)
    if q_i_max > 1e-10:
        q_i_norm = q_i / q_i_max
    else:
        q_i_norm = np.zeros_like(q_i)
    
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Blend QI and EI
    # Weight more towards QI when we want exploration
    if f_best > 10.0:
        qi_weight = 0.7
        ei_weight = 0.3
    else:
        qi_weight = 0.5
        ei_weight = 0.5
    
    score = qi_weight * q_i_norm + ei_weight * ei_norm
    
    # Cost-awareness: divide by cost to prefer cheaper evaluations
    cost_factor = np.sqrt(cost)
    score = score / cost_factor
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        # Return quickly if taking too long
        pass
    
    return score
# EVOLVE-BLOCK-END
