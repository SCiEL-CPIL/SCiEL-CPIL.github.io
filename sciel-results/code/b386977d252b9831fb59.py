"""Cost-aware acquisition function for Bayesian optimization (powell 4d) - Adaptive Cost-Aware UCB."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Precompute constants
_EPS = 1e-9
_DEADLINE_MARGIN = 5  # seconds to stop before the 330s wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    """
    Adaptive Cost-Aware UCB acquisition function.
    
    Strategy:
    1. Compute a base UCB score: -mu + kappa * sigma.
    2. Adapt kappa based on the global uncertainty (median sigma) relative to the 
       spread of the mean predictions (std mu). This allows the algorithm to 
       dynamically shift between exploration (high kappa when uncertain) and 
       exploitation (low kappa when confident).
    3. Normalize by the evaluation cost to prioritize cheap improvements.
    
    This avoids the scale-mismatch issues of averaging heterogeneous metrics 
    (like EI and PI) and provides a single, coherent signal.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])

    # Deadline guard
    t0 = time.time()
    
    # 1. Compute adaptive kappa
    # We want kappa to be large when the model is uncertain (high median sigma)
    # and small when the model is confident (low median sigma or low variance in mu).
    # A common heuristic is to scale kappa with the uncertainty.
    
    # Use median for robustness against outliers in sigma
    med_sigma = np.median(sigma)
    
    # Measure the "spread" of the mean predictions. 
    # If mu is very flat, we should explore more? Or if mu is varied, we exploit?
    # Actually, if std(mu) is small, the landscape is flat, so we should explore (higher kappa).
    # If std(mu) is large, there is structure, so we can exploit (lower kappa).
    # Let's use a ratio: kappa = base * (med_sigma / (std_mu + eps))
    # But we also need to account for the scale of the objective.
    
    std_mu = np.std(mu)
    if std_mu < _EPS:
        std_mu = _EPS
        
    # Base kappa around 1.0 to 2.0
    # If med_sigma is high relative to std_mu, we are in an uncertain regime -> explore.
    # If med_sigma is low, we are in a confident regime -> exploit.
    
    # Normalize the uncertainty ratio to keep kappa in a reasonable range [0.5, 5.0]
    uncertainty_ratio = med_sigma / std_mu
    # Clip the ratio to avoid extreme values
    uncertainty_ratio = np.clip(uncertainty_ratio, 0.1, 10.0)
    
    # Map the ratio to a kappa value. 
    # Linear mapping: if ratio=1, kappa=1.5. If ratio=10, kappa=4.0. If ratio=0.1, kappa=0.5.
    # kappa = 1.5 + (uncertainty_ratio - 1.0) * 0.3
    # Let's try a simpler power law: kappa = 1.0 + 0.5 * uncertainty_ratio
    # This ensures kappa >= 1.0 (always some exploration) and scales with uncertainty.
    kappa = 1.0 + 0.5 * uncertainty_ratio
    
    # Check deadline
    if time.time() - t0 > _DEADLINE_MARGIN:
        # Fallback to simple UCB with fixed kappa=1.0
        cost_safe = np.maximum(cost, _EPS)
        return (-mu + 1.0 * sigma) / cost_safe

    # 2. Compute the adaptive UCB score
    # For minimization: we want high values of -mu + kappa * sigma
    # sigma is already a standard deviation, so it's positive.
    # We ensure sigma is non-negative and safe for division later.
    sigma_safe = np.maximum(sigma, _EPS)
    
    ucb_score = -mu + kappa * sigma_safe
    
    # 3. Cost-awareness
    # Divide by cost. The cost is 1 + mean(x_scaled). 
    # This favors points with high potential improvement per unit cost.
    cost_safe = np.maximum(cost, _EPS)
    
    final_score = ucb_score / cost_safe
    
    # Ensure finite values
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    # Final deadline check
    if time.time() - t0 > _DEADLINE_MARGIN:
        return final_score
        
    return final_score

# EVOLVE-BLOCK-END
