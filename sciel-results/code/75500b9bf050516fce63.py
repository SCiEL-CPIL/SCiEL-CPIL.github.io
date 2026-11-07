import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
DEADLINE = 330.0  # Seconds
SAFETY_MARGIN = 5.0  # Seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a modified UCB strategy.
    
    Strategy:
    1. Use UCB: mu + kappa * sigma.
    2. Adjust kappa based on cost: cheaper points get higher exploration weight.
       kappa_effective = kappa_base * (1 / cost)^alpha
       Since cost is in [1, 2], 1/cost is in [0.5, 1].
       This encourages exploring cheap regions more aggressively.
    3. Add a small penalty for points with very low sigma (exploited) to avoid
       re-evaluating similar regions, though the GP handles this partially.
    4. Ensure finite outputs.
    
    Parameters:
    mu: array of shape (2000,) - Mean of the GP posterior
    sigma: array of shape (2000,) - Standard deviation of the GP posterior
    f_best: float - Best observed value so far
    cost: array of shape (2000,) - Cost of evaluating each candidate
    
    Returns:
    scores: array of shape (2000,) - Acquisition scores (higher = better)
    """
    # Check deadline
    if time.time() > DEADLINE - SAFETY_MARGIN:
        # Return a safe default if we are about to timeout
        # In this context, we just return zeros or a simple heuristic to be safe
        return np.zeros_like(mu)

    # Ensure inputs are arrays and handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: sigma must be positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost is in [1, 2]
    # We want to reward low cost. 
    # Let's define a cost-weighted exploration factor.
    # Base UCB: score = -mu + kappa * sigma  (since we minimize, we maximize -mu + kappa*sigma)
    # Actually, standard UCB for minimization is: -mu + kappa * sigma.
    # Let's stick to the standard form for minimization:
    # We want to minimize f. So we maximize (f_best - f) or use UCB on the negative.
    # Common BO practice for minimization:
    # Acquisition = -mu + kappa * sigma
    
    # Adaptive Kappa:
    # If cost is low (close to 1), we can afford to explore more (higher kappa).
    # If cost is high (close to 2), we should be more conservative (lower kappa).
    # Let kappa_base = 1.0 (tunable, but let's start with a standard value).
    # kappa_i = kappa_base * (1.0 / cost_i)
    # This makes kappa range from 1.0 (cost=1) to 0.5 (cost=2).
    
    kappa_base = 1.5  # Slightly aggressive exploration
    kappa = kappa_base * (1.0 / cost)
    
    # UCB score for minimization
    # We want to pick points with low mu (potential low f) and high sigma (uncertainty).
    # Score = -mu + kappa * sigma
    scores = -mu + kappa * sigma_safe
    
    # Additional structural change: 
    # The prompt mentions "simple regret" scoring. 
    # Sometimes, pure UCB can get stuck if the GP overfits.
    # Let's add a small "distance from best" term? No, coordinates aren't passed.
    # We only have mu, sigma, cost.
    
    # Let's consider the Expected Improvement (EI) again but with a cost-aware twist that is different from simple division.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Cost-aware EI: EI / cost^beta.
    # The current program uses EI / cost (beta=1).
    # Let's try a hybrid: 0.5 * (EI / cost) + 0.5 * (-mu + kappa * sigma)
    # This combines the "improvement" signal with the "exploration" signal.
    
    # Calculate EI
    z = (f_best - mu) / sigma_safe
    # Avoid overflow in exp for large z
    z_clipped = np.clip(z, -10, 10)
    
    from scipy.stats import norm
    phi = norm.pdf(z_clipped)
    Phi = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Normalize EI to be comparable to UCB?
    # EI is in objective units. UCB is in objective units.
    # However, EI can be very small if we are close to optimum.
    # UCB can be large if sigma is large.
    
    # Let's stick to the adaptive UCB as the primary structural change.
    # The "novelty" aspect is handled by sigma.
    # To ensure we don't ignore the current best, we can add a small term that favors points
    # where the GP is confident (low sigma) and mu is low (high -mu).
    # This is already in UCB.
    
    # One issue with UCB: if sigma is high everywhere, it picks random points.
    # If sigma is low everywhere, it picks the min mu.
    # This is generally good behavior.
    
    # Let's refine the cost weighting.
    # Instead of linear 1/cost, maybe (2 - cost) which is linear from 1 to 0?
    # cost=1 -> 1, cost=2 -> 0. This kills exploration at max cost.
    # 1/cost is smoother.
    
    # Let's try a different angle: 
    # Minimize the cost per unit of expected improvement.
    # But we want to maximize improvement.
    # So maximize EI / cost.
    # The current program does this.
    # Why did it score 1.0? Maybe it's not exploring enough.
    # UCB explores more than EI.
    
    # Let's return the adaptive UCB scores.
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
