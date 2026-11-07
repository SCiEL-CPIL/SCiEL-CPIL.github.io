"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a smooth cost penalty. Uses EI/cost^beta formulation with
    adaptive exploration via UCB-like term.
    
    The key structural change: instead of hard gating, we use a smooth
    cost-normalized acquisition that balances:
    1. Expected improvement (exploitation)
    2. Uncertainty (exploration)
    3. Cost efficiency (cost-awareness)
    
    We compute a composite score that favors cheap points with high
    potential for improvement, while still exploring uncertain regions.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (standard)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization: cost ranges from 1 to 2 (1 + mean of two coords in [0,1])
    # We want to strongly prefer cheap points
    # Use a power law: score = (ei + exploration) / cost^alpha
    
    # Adaptive alpha: as budget depletes (more evaluations), increase cost sensitivity
    # cost here is the per-candidate evaluation cost
    # We can't know total budget used, but we can infer from f_best magnitude
    # Lower f_best suggests more progress, so be more exploitative
    # Actually, let's use a fixed but strong cost penalty
    
    alpha = 1.5  # cost sensitivity exponent
    
    # Exploration term: upper confidence bound component
    # Beta controls exploration vs exploitation
    # Use a moderate beta that adapts: more exploration when sigma is high
    beta = 1.0
    
    ucb = mu + beta * sigma_safe
    
    # Combine: we want to minimize, so improvement is when mu is low
    # The acquisition should be high when we expect good (low) values at low cost
    # Score = (expected improvement + exploration bonus) / cost^alpha
    
    # Exploration bonus: how much uncertainty could lead to improvement
    # If sigma is large and mu is not terrible, there's potential
    exploration_bonus = beta * sigma_safe
    
    # Composite numerator: combine EI and exploration
    numerator = ei + 0.3 * exploration_bonus
    
    # Cost penalty
    cost_safe = np.maximum(cost, 1e-8)
    cost_penalty = np.power(cost_safe, alpha)
    
    # Final score
    score = numerator / cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
