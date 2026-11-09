"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 300.0  # 300s limit per evaluation, leave margin

def _check_deadline():
    """Return True if we should stop and return current best."""
    return (time.time() - _START_TIME) > (_DEADLINE_SECONDS - 5.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement (EI)
    and Probability of Improvement (PoI), with adaptive weighting based on
    uncertainty and cost.
    
    This differs from LCB by using a probabilistic framework that explicitly
    computes the expected gain over the current best, which tends to be more
    robust to noisy GP predictions. The adaptive combination with PoI helps
    in early stages when uncertainty is high.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions in objective units
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best (lowest) observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition values (higher = better to evaluate)
    """
    # Ensure inputs are numpy arrays and handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute Probability of Improvement (PoI)
    # PoI = Phi((f_best - mu)/sigma)
    poi = norm.cdf(z)
    
    # Adaptive weighting:
    # In early stages (high uncertainty relative to regret), PoI is more useful
    # as it identifies regions with high probability of being good.
    # In later stages (low uncertainty, close to optimum), EI is more precise.
    
    # Regret relative to global min
    GLOBAL_MIN = -10.5364
    regret = max(0.0, f_best - GLOBAL_MIN)
    
    # Normalized uncertainty: how large is sigma relative to the scale of the problem
    # Use mean sigma as a proxy for overall uncertainty level
    mean_sigma = np.mean(sigma_safe)
    
    # When regret is large and uncertainty is high, blend more PoI
    # When regret is small or uncertainty is low, rely more on EI
    # Weight for PoI: increases with regret and mean_sigma
    poi_weight = 0.3 + 0.2 * min(regret / 5.0, 1.0) + 0.2 * min(mean_sigma / 2.0, 1.0)
    poi_weight = np.clip(poi_weight, 0.0, 1.0)
    
    # Blend EI and PoI
    # Normalize both to have similar scales before blending
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    combined = (1 - poi_weight) * ei_norm + poi_weight * poi
    
    # Cost-awareness: divide by cost to prefer cheaper evaluations
    scores = combined / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
