# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline tracking
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) with an adaptive multiplicative cost penalty.
    
    Parameters:
    - mu: mean of GP predictions, shape (2000,)
    - sigma: std dev of GP predictions, shape (2000,)
    - f_best: best observed function value
    - cost: cost of evaluating each candidate, shape (2000,)
    
    Returns:
    - scores: acquisition values, shape (2000,)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Deadline guard: if we are running out of time, return a simple heuristic
    # to ensure we don't get killed.
    # The budget is 300s. We check if we have exceeded 290s.
    if _get_elapsed() > 290:
        # Fallback: just pick the point with lowest mean, ignoring cost and uncertainty
        # This is O(N) and very fast.
        scores = -mu
        return np.where(np.isfinite(scores), scores, -np.inf)

    # Clip sigma to avoid numerical issues (division by zero)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero or negative costs
    # Cost is defined as 1 + mean(x scaled to [0,1]), so it's in [1, 2].
    # But we ensure it's at least 1e-9 for safety.
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # Improvement is max(0, f_best - f(x))
    # Let z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is the CDF of the standard normal distribution
    # phi is the PDF of the standard normal distribution
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    EI = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Adaptive Cost Penalty
    # We want to penalize cost more as we get closer to the optimum (f_best decreases).
    # gamma controls the strength of the penalty.
    # gamma = gamma_max * exp(-f_best / scale)
    # If f_best is large (e.g., 5), exp(-5) is small, so gamma is small -> weak penalty.
    # If f_best is small (e.g., 0.1), exp(-0.1) is close to 1, so gamma is large -> strong penalty.
    
    gamma_max = 1.5
    scale = 1.0
    
    # Clip f_best to avoid extreme values causing exp underflow/overflow issues
    # f_best is typically positive for this function.
    f_best_clipped = np.clip(f_best, 0, 10)
    
    gamma = gamma_max * np.exp(-f_best_clipped / scale)
    
    # Apply multiplicative penalty: Score = EI / (cost^gamma)
    # This is equivalent to EI * cost^(-gamma)
    # Using log-space for numerical stability?
    # log_score = log(EI) - gamma * log(cost)
    # But EI can be 0 or negative? No, EI is always >= 0.
    # However, if EI is 0, log is -inf.
    # Let's just compute directly. N=2000 is small.
    
    # Handle cases where EI is 0
    # If EI is 0, score should be 0 (or very small).
    
    # To avoid issues with cost^gamma if cost is large, we can use:
    # penalty = np.power(cost_safe, gamma)
    # scores = EI / penalty
    
    # A safer way if EI can be very small:
    # scores = EI * np.exp(-gamma * np.log(cost_safe))
    
    log_cost = np.log(cost_safe)
    scores = EI * np.exp(-gamma * log_cost)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # If all scores are 0 (e.g., f_best is very low and EI is negligible everywhere),
    # fallback to a small random noise or just -mu to break ties.
    if np.all(scores == 0.0):
        scores = np.random.rand(len(mu)) * 1e-6 - mu * 1e-6
        
    return scores
# EVOLVE-BLOCK-END
