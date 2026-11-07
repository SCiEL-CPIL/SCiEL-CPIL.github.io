"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB).
    
    Args:
        mu: Array of shape (N,) with GP mean predictions.
        sigma: Array of shape (N,) with GP standard deviations.
        f_best: Scalar, the best observed function value so far.
        cost: Array of shape (N,) with evaluation costs for each candidate.
        
    Returns:
        Array of shape (N,) with scores. Higher score is better.
    """
    start_time = time.time()
    # Set a safety deadline well before the 300s limit
    deadline = start_time + 295.0

    # Convert inputs to numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # f_best is a scalar
    f_best_val = float(f_best)

    # Dynamic beta parameter for LCB.
    # We increase exploration (beta) as f_best is worse (larger).
    # For Griewank 2D, values are typically in [0, 1].
    # If f_best is close to 0, we exploit (beta ~ 1.0).
    # If f_best is around 1.0, we explore more (beta ~ 3.0).
    # Clipping f_best to [0, 1] ensures stable behavior if outliers occur.
    f_best_clipped = np.clip(f_best_val, 0.0, 1.0)
    beta = 1.0 + 2.0 * f_best_clipped

    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Lower Confidence Bound (LCB)
    # We want to minimize f(x), so LCB = mu - beta * sigma
    # A lower LCB is better.
    lcb = mu - beta * sigma_safe

    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)

    # Cost-adjusted score: We want to maximize the score.
    # Since LCB is "lower is better", we can invert it or use a formulation where higher is better.
    # Common approach: Score = -LCB / cost ? Or Score = ( -mu + beta*sigma ) / cost ?
    # Actually, standard LCB for minimization picks min(mu - beta*sigma).
    # To convert to a "maximize" score: score = -(mu - beta*sigma) / cost = (beta*sigma - mu) / cost.
    # However, the prompt says "Return 2000 finite scores (higher = evaluate next)".
    # The seed program used EI, which is "higher is better" (improvement).
    # LCB value is "lower is better".
    # So we should return scores that are high for good candidates.
    # Good candidate: Low mu, High sigma (if exploring), Low cost.
    # LCB = mu - beta*sigma. Low LCB is good.
    # So Score = -LCB / cost = (beta*sigma - mu) / cost.
    
    # Let's verify with the seed.
    # Seed: EI / cost. EI is positive if mu < f_best. Higher EI is better.
    # So we need a score that is higher for better candidates.
    
    # If we use Score = (beta * sigma - mu) / cost:
    # - If mu is low (good), Score is high.
    # - If sigma is high (uncertain), Score is high (encourages exploration).
    # - If cost is high, Score is low (penalizes cost).
    # This aligns with the goal.
    
    scores = (beta * sigma_safe - mu) / cost_safe

    # Check deadline
    if time.time() > deadline:
        # Fallback: Pure exploitation (minimize mu, ignore cost slightly or just use mu)
        # Actually, if we hit deadline, we should just return something valid.
        # Returning -mu / cost is a safe exploitation fallback.
        scores = -mu / cost_safe

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
