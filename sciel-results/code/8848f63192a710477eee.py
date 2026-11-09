"""Cost-aware LCB acquisition for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_t0 = time.time()
_DEADLINE = 320.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Score = -(mu + kappa * sigma) / (1 + alpha * (cost - cost_min))
    Higher score = more likely to be evaluated next.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate, in [1, 2].
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard
    if time.time() - _t0 > _DEADLINE:
        # Fallback: exploit best known, penalize cost
        return -mu / (1.0 + 0.5 * cost)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Clamp sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)

    # Adaptive kappa: exploration strength
    # When f_best is far from 0 (assuming min is near 0 for most functions),
    # we need more exploration. As f_best approaches the minimum, exploit.
    # Use a smooth schedule based on |f_best| relative to a scale.
    # The functions have minima at 0 (ackley, rastrigin, levy) or -3.86 (hartmann).
    # We don't know which function, so use f_best itself as a proxy.
    # Early: f_best is large (positive for most), later: f_best -> min.
    # Scale: use max(|f_best|, 1.0) to avoid division by zero.
    f_scale = max(abs(f_best), 1.0)
    # kappa starts high (explore) and decays as we get closer to minimum
    # Use a soft decay: kappa = base / (1 + progress)
    # progress is roughly (initial_f - f_best) / initial_f, but we don't know initial_f.
    # Simpler: kappa = 2.0 / (1.0 + abs(f_best) * 0.1)
    # When f_best is large (early), kappa ~ 2.0/(1+large) ~ small... that's wrong.
    # Let's think: early on f_best is bad (large for min-at-0 functions).
    # We want high kappa early, low kappa late.
    # So kappa should decrease as f_best decreases (for min-at-0) or increases (for hartmann).
    # Since we don't know the direction, use a fixed moderate kappa with slight adaptation.
    # Alternative: use the ratio of sigma to mu to gauge uncertainty.
    
    # Actually, let's use a simpler adaptive scheme:
    # kappa = 1.5 + 1.0 * exp(-abs(f_best) / 2.0)
    # When f_best is near 0 (good), exp(0)=1, kappa=2.5 (more exploration near optimum to find it)
    # When f_best is large (bad), exp(-large)~0, kappa=1.5 (less exploration, just find good points)
    # Hmm, that's backwards. When f_best is bad, we should explore MORE to find good points.
    
    # Let's reconsider: 
    # - f_best is bad (far from min): we need to explore broadly. High kappa.
    # - f_best is good (near min): we can exploit. Low kappa.
    # For min-at-0 functions: f_best starts positive, decreases to 0.
    #   So as f_best decreases, we want kappa to decrease.
    #   kappa = base * (1 + f_best / scale) if f_best > 0
    # For hartmann: f_best starts positive (or less negative), decreases to -3.86.
    #   So f_best decreases, we want kappa to decrease.
    # In both cases, f_best decreases toward the minimum.
    # So: kappa should decrease as f_best decreases.
    #   kappa = base * max(1.0, f_best / 1.0)  -- but f_best can be negative.
    
    # Simpler robust approach: use a fixed kappa that works well in practice.
    # LCB with kappa ~ 1.0-2.0 is standard. Let's use kappa = 1.5 as a solid default.
    # The cost-awareness provides the other axis of the tradeoff.
    
    kappa = 1.5
    
    # Cost penalty: normalize cost to [0, 1] range for smoother penalty
    # cost is in [1, 2], so cost - 1 is in [0, 1]
    cost_norm = cost - 1.0  # in [0, 1]
    
    # Cost-aware denominator: 1 + alpha * cost_norm
    # alpha controls how strongly we penalize expensive evaluations
    # alpha = 1.0 means a cost-2 point gets half the score of a cost-1 point (all else equal)
    alpha = 1.0
    
    denom = 1.0 + alpha * cost_norm
    
    # LCB score: lower mu + kappa*sigma is better (more promising)
    # We negate so higher score = better
    lcb = -(mu + kappa * sigma)
    
    # Divide by cost denominator to penalize expensive evaluations
    scores = lcb / denom
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
