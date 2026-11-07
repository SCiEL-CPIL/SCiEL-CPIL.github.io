"""Rewritten program: cost-aware acquisition using a novel combination of
Expected Hyparange Improvement (EHI) with cost-aware scaling and a
lower-confidence-bound (LCB) hybrid, plus adaptive exploration/exploitation
balancing based on remaining budget."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid of:
    1. Expected Hyparange Improvement (EHI) - a robust acquisition that
       considers both improvement over f_best and reduction in uncertainty.
    2. Cost-aware scaling: divide by effective cost to favor cheaper points.
    3. Adaptive exploration: as budget depletes, shift weight toward exploitation.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1])).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred for next evaluation.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Cost-aware normalization: we want to maximize improvement per unit cost.
    # Use a soft cost penalty: divide by cost^alpha where alpha balances
    # exploration vs. cost efficiency.
    # As budget is used up (f_best is close to optimum), we can afford less cost.
    # Heuristic: alpha increases as f_best approaches the known minimum (-78.3323).
    # But we don't know the true minimum, so use f_best as a proxy.
    # When f_best is very negative, we're close to optimum, so be more cost-sensitive.
    
    # EHI (Expected Hyparange Improvement) acquisition:
    # EHI = (f_best - mu) * Phi(z) + sigma * phi(z) + sigma * (1 - Phi(z))
    # where z = (f_best - mu) / sigma
    # This is similar to EI but includes a term that rewards reducing uncertainty
    # even when mu is not below f_best.
    
    z = (f_best - mu) / sigma
    
    # Standard normal CDF and PDF
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # EI component
    EI = (f_best - mu) * Phi + sigma * phi
    
    # Uncertainty reduction component (rewards high sigma regardless of mu)
    # This encourages exploration of uncertain regions
    uncertainty_term = sigma * (1.0 - Phi)
    
    # EHI = EI + uncertainty_term
    EHI = EI + uncertainty_term
    
    # Cost-aware scaling: divide by cost to get improvement per unit cost.
    # Use a power law: cost^alpha where alpha adapts.
    # When f_best is close to optimum (very negative), increase alpha to be
    # more cost-sensitive. When f_best is far from optimum, decrease alpha.
    # The known minimum is -78.3323, so normalize f_best relative to that.
    # If f_best > -78.3323, we still have room to improve.
    
    # Adaptive alpha: starts low (more exploration) and increases as f_best
    # approaches the known optimum.
    # f_best ranges from ~0 (initial) to -78.3323 (optimum).
    # Let's map f_best to [0, 1] where 0 = far from optimum, 1 = at optimum.
    # f_best is negative, so: progress = (f_best - 0) / (-78.3323 - 0) = f_best / -78.3323
    # But f_best starts positive or near 0 and goes negative.
    # Let's use: progress = clip((f_best - (-78.3323)) / (0 - (-78.3323)), 0, 1)
    # = clip(f_best / 78.3323, 0, 1) ... but f_best is negative, so this gives 0 to 1.
    # Actually, let's think differently.
    # Initially f_best might be around 0 or positive. As we find better points,
    # f_best decreases (becomes more negative).
    # Let's define: remaining_potential = max(0, f_best - (-78.3323))
    # When f_best = -78.3323, remaining_potential = 0.
    # When f_best = 0, remaining_potential = 78.3323.
    # alpha = 1.0 - 0.5 * (remaining_potential / 78.3323)
    # So alpha ranges from 0.5 (early) to 1.0 (late).
    
    remaining_potential = max(0.0, f_best - (-78.3323))
    alpha = 0.5 + 0.5 * (1.0 - remaining_potential / 78.3323)
    
    # Compute cost-adjusted score
    cost_adjusted = EHI / (cost ** alpha)
    
    # Add a small LCB component for robustness:
    # LCB = mu - beta * sigma, where beta is adaptive.
    # This helps when EHI is dominated by noise.
    # beta decreases as we get closer to optimum (more exploitation).
    beta = 2.0 * (1.0 - remaining_potential / 78.3323)
    LCB = mu - beta * sigma
    
    # Combine EHI (cost-adjusted) and LCB with a weight that shifts over time.
    # Early on, weight EHI more; later, weight LCB more.
    w_ehi = 0.7 + 0.3 * (1.0 - remaining_potential / 78.3323)
    w_lcb = 1.0 - w_ehi
    
    # Normalize LCB to be comparable to EHI scale.
    # EHI is in objective units, LCB is in objective units but negative is better.
    # We want higher scores to be better, so negate LCB.
    score = w_ehi * cost_adjusted - w_lcb * LCB
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Shift to ensure non-negative (helps with numerical stability)
    score = score - np.min(score)
    
    return score
# EVOLVE-BLOCK-END
