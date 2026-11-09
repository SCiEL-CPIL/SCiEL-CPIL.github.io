"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization (minimization).
    
    Combines Expected Improvement (EI) with a cost-aware linear penalty and an
    auto-calibrated exploration bonus. The structure is:
    
    score(x) = EI(x) - lambda_cost * cost(x) + lambda_exp * sigma(x)
    
    where EI is the analytic Expected Improvement for minimization, and the
    lambda parameters are auto-calibrated from the surrogate statistics so that
    cost and uncertainty act as proportional tie-breakers rather than dominant
    terms. This avoids the over-penalization of expensive regions that pure
    EI/cost division introduces, while still guiding the search toward cheap,
    high-improvement candidates.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable).
    """
    t_start = time.time()
    deadline = 290.0  # Safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n = mu.size

    # Numerical guards
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 1e-10), sigma, 1e-6)
    cost_safe = np.where(np.isfinite(cost) & (cost > 1e-10), cost, 1.0)
    sigma_safe = np.maximum(sigma_safe, 1e-8)
    cost_safe = np.maximum(cost_safe, 1e-8)

    # Compute Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For GP: f(x) ~ N(mu, sigma^2)
    improvement = f_best - mu
    sigma_for_ei = np.maximum(sigma_safe, 1e-10)
    
    z = improvement / sigma_for_ei
    z = np.clip(z, -40.0, 40.0)
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = improvement * Phi + sigma_for_ei * phi
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Auto-calibrate weights based on current surrogate statistics
    # We want the cost penalty and exploration bonus to be small perturbations
    # relative to the typical EI, so that EI dominates but cost/uncertainty
    # break ties and guide the search.
    
    # Estimate typical scales using medians for robustness against outliers
    finite_ei = ei[np.isfinite(ei) & (ei > 1e-12)]
    if finite_ei.size > 0:
        scale_ei = float(np.median(finite_ei))
    else:
        # If no improvement is expected, fall back to sigma scale
        finite_sig = sigma_safe[np.isfinite(sigma_safe)]
        scale_ei = float(np.median(finite_sig)) if finite_sig.size else 1.0
    if not np.isfinite(scale_ei) or scale_ei < 1e-8:
        scale_ei = 1.0

    finite_cost = cost_safe[np.isfinite(cost_safe)]
    scale_cost = float(np.median(finite_cost)) if finite_cost.size else 1.0
    if not np.isfinite(scale_cost) or scale_cost < 1e-8:
        scale_cost = 1.0

    finite_sig = sigma_safe[np.isfinite(sigma_safe)]
    scale_sigma = float(np.median(finite_sig)) if finite_sig.size else 1.0
    if not np.isfinite(scale_sigma) or scale_sigma < 1e-8:
        scale_sigma = 1.0

    # Set lambda_cost such that lambda_cost * scale_cost = 0.1 * scale_ei
    # This makes cost a 10% tie-breaker factor
    lambda_cost = 0.1 * scale_ei / scale_cost
    
    # Set lambda_exp such that lambda_exp * scale_sigma = 0.1 * scale_ei
    # This makes exploration a 10% tie-breaker factor
    lambda_exp = 0.1 * scale_ei / scale_sigma

    # Ensure lambdas are finite and non-negative
    lambda_cost = float(np.clip(lambda_cost, 0.0, 1e6))
    lambda_exp = float(np.clip(lambda_exp, 0.0, 1e6))

    # Construct the final score
    # Higher score is better
    # EI is the primary driver
    # Cost is penalized
    # Uncertainty is rewarded
    scores = ei - lambda_cost * cost_safe + lambda_exp * sigma_safe

    # Ensure all values are finite (higher = evaluate next)
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard (safety margin before the 300s kill)
    if (time.time() - t_start) > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
