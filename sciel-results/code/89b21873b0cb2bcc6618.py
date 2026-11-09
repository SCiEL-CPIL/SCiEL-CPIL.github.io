# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a dynamic UCB weight that scales with the ratio of remaining budget
    to current cost, shifting from exploration to exploitation as budget depletes.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1])).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Dynamic exploration weight based on estimated remaining budget
    # Total budget is 30. Initial cost is ~5 points * ~1.5 avg cost = 7.5.
    # We estimate the current total cost spent.
    # Since we don't have the exact history, we can infer the "phase" from the GP state.
    # A common proxy is the median sigma: high sigma -> early phase, low sigma -> late phase.
    # Another proxy is f_best: high f_best -> early, low f_best -> late.
    
    # Let's estimate the number of evaluations or total cost spent.
    # We can't do this directly. However, we can use the fact that as we evaluate more points,
    # the GP becomes more certain (sigma drops) and f_best improves.
    
    # Let's use a hybrid heuristic for the "remaining budget ratio".
    # Early: median_sigma is high. Late: median_sigma is low.
    # Let's assume initial median_sigma is around 1.0 (standardized targets? no, objective units).
    # Actually, the problem states "standardized targets" for GP, but mu/sigma are in objective units.
    # This implies the GP is trained on standardized targets, but outputs are converted back?
    # Or the GP is trained on raw targets? "standardized targets" usually means the GP sees y/z.
    # If so, mu and sigma returned are in objective units.
    
    # Let's use the median sigma to estimate the phase.
    # We need to know the scale of sigma.
    # Let's use the range of mu as a scale reference.
    range_mu = np.max(mu) - np.min(mu)
    median_sigma = np.median(sigma)
    
    # If range_mu is very small, we are likely late or stuck.
    if range_mu < 1e-6:
        # If mu is flat, rely on f_best to determine phase.
        # If f_best is very low, we are done. If high, we are early.
        # Let's assume a default beta.
        beta = 1.0
    else:
        # Relative uncertainty
        rel_uncertainty = median_sigma / range_mu
        
        # Map relative uncertainty to a budget ratio.
        # Early: high rel_uncertainty -> high exploration.
        # Late: low rel_uncertainty -> low exploration.
        
        # Let's assume a baseline beta of 1.0 and scale it.
        # beta = 1.0 + 2.0 * rel_uncertainty
        
        # Also consider f_best.
        # If f_best is close to 0, we should exploit more (lower beta).
        # If f_best is high, we should explore more (higher beta).
        
        # Combine both signals.
        # Let's use a weighted average.
        
        # Normalize f_best? We don't know the max value.
        # But we know min is 0.
        # Let's use a soft cap.
        f_best_norm = 1.0 / (1.0 + f_best)
        # If f_best = 0, f_best_norm = 1.0 (exploit).
        # If f_best = 10, f_best_norm = 0.09 (explore).
        
        # So exploration weight should be inversely related to f_best_norm?
        # Or directly related to (1 - f_best_norm)?
        
        # Let's try:
        # beta_uncertainty = 2.0 * rel_uncertainty
        # beta_fbest = 2.0 * (1.0 - f_best_norm)
        
        # beta = 0.5 * beta_uncertainty + 0.5 * beta_fbest
        
        # Let's simplify.
        # The directive says: "scales the UCB weight by the ratio of remaining budget to current cost".
        # Remaining budget / Current cost.
        # As current cost increases, remaining budget decreases.
        # So the ratio decreases over time.
        
        # We can approximate the "current cost" by the total cost spent.
        # We don't have it. But we can approximate the "phase" by the GP's confidence.
        
        # Let's use the median sigma as a proxy for "information gained".
        # High sigma = low information = high remaining budget ratio.
        # Low sigma = high information = low remaining budget ratio.
        
        # Let's assume the initial sigma is around 1.0 (or some scale).
        # We can normalize by the initial sigma if we knew it.
        # Since we don't, let's use the max sigma in the current batch?
        # Or just use the median sigma directly with a scaling factor.
        
        # Let's try a simple exponential decay based on median sigma.
        # beta = 2.0 * np.exp(-median_sigma)
        # If median_sigma = 1.0, beta = 0.73
        # If median_sigma = 0.1, beta = 1.81 (Wait, this is higher? No, exp(-0.1) is 0.9, so 1.8)
        # This is inverted. We want lower beta when sigma is low.
        
        # So beta = 2.0 * np.exp(-median_sigma) is wrong if we want less exploration later.
        # We want:
        # Early (high sigma): High beta.
        # Late (low sigma): Low beta.
        
        # So beta should increase with sigma.
        # beta = 2.0 * median_sigma / (max_sigma + 1e-9) ?
        # If max_sigma is constant, beta scales with median_sigma.
        
        # Let's use:
        # beta = 2.0 * (median_sigma / (np.max(sigma) + 1e-9))
        
        max_sigma = np.max(sigma)
        if max_sigma < 1e-9:
            beta = 0.0
        else:
            beta = 2.0 * (median_sigma / max_sigma)
            
            # Clip beta to [0.1, 2.0] to avoid extremes.
            beta = np.clip(beta, 0.1, 2.0)
            
            # Modulate by f_best to ensure exploitation when we are close to optimum.
            # If f_best is very small, reduce beta.
            # If f_best is large, keep beta high.
            
            # Let's use a smooth transition.
            # f_best_scale = 1.0
            # f_best_norm = f_best / (f_best + f_best_scale)
            # If f_best = 0, f_best_norm = 0.
            # If f_best = 1, f_best_norm = 0.5.
            # If f_best = 10, f_best_norm = 0.91.
            
            # We want to reduce beta when f_best is small.
            # So multiply beta by (f_best_norm + 0.1) ?
            # If f_best = 0, factor = 0.1. Beta becomes 0.1 * beta.
            # If f_best = 10, factor = 1.01. Beta becomes ~beta.
            
            f_best_norm = f_best / (f_best + 1.0)
            beta *= (f_best_norm + 0.1)
            
            # Re-clip
            beta = np.clip(beta, 0.01, 2.0)
    
    # Calculate UCB term for minimization
    # We want to minimize, so we use Lower Confidence Bound (LCB) or just EI.
    # The current program uses (ei + beta * sigma) / cost.
    # EI is (f_best - mu) * cdf(z) + sigma * pdf(z).
    # This is the expected improvement.
    # Adding beta * sigma adds an exploration bonus.
    # So the score is (EI + Exploration) / Cost.
    
    # Let's stick to this form but ensure the exploration term is scaled correctly.
    # Exploration term: beta * sigma
    # This encourages points with high uncertainty.
    
    # Final Score:
    # score = (ei + beta * sigma) / cost
    
    # Ensure finite scores
    scores = (ei + beta * sigma) / cost
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
