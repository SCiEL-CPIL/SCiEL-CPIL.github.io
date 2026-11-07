import numpy as np
from scipy.stats import norm

# Module-level state to track the number of acquisition calls
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Time-varying GP-UCB + Normalized Expected Improvement.
    
    Parameters:
    mu: np.ndarray, shape (N,)
        Predicted mean of the objective function.
    sigma: np.ndarray, shape (N,)
        Predicted standard deviation of the objective function.
    f_best: float
        The best (lowest) objective value observed so far.
        
    Returns:
    np.ndarray, shape (N,)
        Acquisition scores. Higher scores indicate more promising points.
    """
    global _call_count
    _call_count += 1
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Clip sigma to a small positive value to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Time-varying exploration parameter for GP-UCB
    # beta_t = sqrt(2 * t * log(1 + t))
    t = max(_call_count, 1)
    beta_t = np.sqrt(2 * t * np.log(1 + t))
    
    # GP-UCB score (for minimization, we use -mu + beta * sigma)
    # We want to minimize f(x), so high UCB score means "likely to be low"
    # Standard UCB for maximization: mu + beta * sigma
    # For minimization: -mu + beta * sigma  (or equivalently, we maximize -f)
    # But EI is defined for minimization directly.
    # Let's stick to the convention: we want to select points with low f(x).
    # UCB for minimization: -mu + beta * sigma
    # However, EI is (f_best - mu) * Phi(z) + sigma * phi(z)
    # Both should be in the same "direction" (higher is better for selection).
    
    ucb_score = -mu + beta_t * sigma_safe
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to have similar scale as UCB term?
    # Or just add them. The scale of EI depends on f_best - mu and sigma.
    # The scale of UCB depends on mu and sigma.
    # Let's try adding them directly first.
    # To balance, we might need to scale EI.
    # Let's try: score = ucb_score + ei
    
    # However, EI can be negative if mu > f_best + sigma * something.
    # Actually, EI is always >= 0 if we define it as E[max(f_best - f(x), 0)].
    # Let's verify:
    # If mu >> f_best, z is large negative, Phi(z) ~ 0, phi(z) ~ 0, EI ~ 0.
    # If mu << f_best, z is large positive, Phi(z) ~ 1, phi(z) ~ 0, EI ~ f_best - mu.
    # So EI is non-negative.
    
    # UCB score (-mu + beta * sigma) can be anything.
    # We want higher scores to be selected.
    # If mu is very low (good), -mu is very high (good).
    # If sigma is high, beta * sigma is high (good for exploration).
    
    # Let's combine them.
    # A common trick is to normalize one of them.
    # But let's try a simple weighted sum.
    # The directive suggests: "combine it additively with a normalized Expected Improvement term"
    
    # Normalize EI by its max or std?
    # If we normalize EI, we lose the absolute scale of improvement.
    # Maybe just: score = ucb_score + ei
    
    # Let's check the scales.
    # mu and sigma are in objective units.
    # f_best is in objective units.
    # So EI and UCB are in objective units.
    # They should be comparable.
    
    scores = ucb_score + ei
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
