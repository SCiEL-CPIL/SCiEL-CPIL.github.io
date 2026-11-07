import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a Knowledge Gradient (KG) style bonus that rewards points likely
    to reveal new basins in the Rastrigin landscape.
    
    The KG bonus is approximated by the probability that the true value
    at x is significantly below f_best (indicating a potentially new
    minimum basin), weighted by the uncertainty. This encourages exploration
    of regions with high uncertainty that could contain lower minima,
    rather than just exploiting the current best.
    
    Parameters:
    mu (np.array): Mean of the Gaussian Process, shape (2000,).
    sigma (np.array): Standard deviation of the Gaussian Process, shape (2000,).
    f_best (float): Best observed value so far.
    cost (np.array): Cost of evaluating each candidate, shape (2000,).
    
    Returns:
    np.array: Acquisition scores, shape (2000,). Higher is better.
    """
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin before 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma and cost are positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f, 0)] where f ~ N(mu, sigma^2)
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = improvement * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Compute KG-style bonus: probability that f(x) is significantly below f_best
    # This rewards points where the GP predicts a value much lower than f_best,
    # suggesting a potentially new basin. We use a threshold based on the 
    # improvement magnitude relative to uncertainty.
    
    # The "knowledge gradient" intuition: evaluate x if it is likely to 
    # change our belief about the global minimum. For Rastrigin with well-separated
    # basins, this means finding points where f(x) << f_best.
    
    # Define a "significant improvement" threshold: f(x) < f_best - threshold
    # where threshold scales with the current best value and uncertainty.
    # When f_best is large (early), we want to find any significant drop.
    # When f_best is small (late), we want to find even smaller improvements.
    
    if f_best > 50.0:
        kg_threshold = 5.0  # Look for drops of at least 5 units
        kg_weight = 0.4
    elif f_best > 10.0:
        kg_threshold = 2.0
        kg_weight = 0.3
    elif f_best > 1.0:
        kg_threshold = 0.5
        kg_weight = 0.2
    else:
        kg_threshold = 0.1
        kg_weight = 0.1
    
    # Probability that f(x) < f_best - kg_threshold
    # P(f < f_best - kg_threshold) = Phi((f_best - kg_threshold - mu) / sigma)
    z_kg = (f_best - kg_threshold - mu) / sigma_safe
    z_kg = np.clip(z_kg, -10, 10)
    prob_significant_improvement = norm.cdf(z_kg)
    
    # KG bonus: probability of significant improvement, scaled by uncertainty
    # Higher uncertainty in a region that could contain a new basin is valuable
    kg_bonus = prob_significant_improvement * sigma_safe
    
    # Combine EI and KG bonus
    # EI is the primary driver, KG bonus provides exploration incentive
    combined_score = ei + kg_weight * kg_bonus
    
    # Cost-awareness: divide by cost to prefer cheaper evaluations
    # Use square root of cost to soften the penalty while still preferring cheap points
    cost_factor = np.sqrt(cost_safe)
    score = combined_score / cost_factor
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Return quickly if taking too long
        pass
    
    return score
