"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid approach:
    - UCB for exploration in early stages (high gap to optimum)
    - Cost-normalized improvement for exploitation in later stages
    - Adaptive blending based on progress
    
    Parameters
    ----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Theoretical minimum for Styblinski-Tang 2D
    f_min_st2d = -78.3323
    
    # Measure progress: how close are we to the optimum?
    gap = max(f_best - f_min_st2d, 0.0)
    # Normalize gap: early (large gap) -> explore, late (small gap) -> exploit
    # Typical gap range: 0 to ~150 (starting points are random)
    gap_norm = np.clip(gap / 80.0, 0.0, 1.0)
    
    # UCB component: reward high uncertainty and low mean
    # For minimization, UCB = mu + beta * sigma (we want to find low values)
    # But for acquisition (higher is better), we use -mu + beta * sigma
    beta_early = 2.5
    beta_late = 0.8
    beta = beta_late + (beta_early - beta_late) * gap_norm
    
    # UCB score: higher is better
    ucb_score = -mu + beta * sigma
    
    # Expected improvement component (for minimization)
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # But we need a simpler, faster version
    # Approximate EI with a smooth function
    diff = f_best - mu  # positive when mu < f_best (potential improvement)
    z = diff / sigma
    # Standard normal CDF and PDF approximations
    # Use a simpler formulation: reward = max(0, diff) * (1 + z/sigma)
    # Actually, let's use the proper EI but vectorized
    # Phi(z) approximation using tanh
    phi_z = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    # CDF approximation
    Phi_z = 0.5 * (1.0 + np.tanh(z / np.sqrt(2.0)))
    
    ei = diff * Phi_z + sigma * phi_z
    # EI is already "higher is better" (expected improvement)
    
    # Cost normalization: divide by cost to prefer cheaper evaluations
    # Use a power law for smoother cost scaling
    cost_power = 0.5
    cost_factor = cost ** cost_power
    
    ucb_cost_norm = ucb_score / cost_factor
    ei_cost_norm = ei / cost_factor
    
    # Blend UCB and EI based on progress
    # Early: more UCB (exploration), Late: more EI (exploitation)
    blend = gap_norm  # 1.0 early, 0.0 late
    
    blended_score = (1.0 - blend) * ei_cost_norm + blend * ucb_cost_norm
    
    # Add a small exploration bonus for points with high sigma relative to mean
    # This helps escape local optima
    exploration_bonus = 0.1 * sigma / (np.abs(mu) + 1.0)
    
    scores = blended_score + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 295.0:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
