"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute constants for speed
_EPS = 1e-12

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using an adaptive schedule based on the progress of the optimization.
    
    Strategy:
    The Weierstrass function in 1D on [-0.5, 0.5] has a global minimum of 0.
    We know the target value is 0.
    
    In the early stages, f_best is likely far from 0 (e.g., > 0.1). We should prioritize
    exploration to find better regions. This corresponds to a lower xi in EI (more
    optimistic) or a higher kappa in UCB.
    
    In the later stages, f_best is close to 0 (e.g., < 0.01). We should prioritize
    exploitation to refine the minimum. This corresponds to a higher xi in EI (more
    conservative) or a lower kappa in UCB.
    
    We use a blend of EI and UCB, but the weights and parameters are adaptive.
    
    Let progress = 1 - (f_best - 0) / (initial_f_best_estimate - 0).
    Since we don't know initial_f_best, we can normalize f_best relative to a reasonable
    upper bound for the Weierstrass function on this domain. The max value is around 1-2.
    Let's assume a typical starting f_best is around 0.5 to 1.0.
    
    A simpler robust approach:
    Use EI with an adaptive xi.
    xi = base_xi * (f_best / reference_value)
    If f_best is large, xi is large? No, we want MORE exploration when f_best is bad.
    Exploration means being willing to pick points with higher expected improvement but
    also potentially higher variance.
    
    Actually, standard EI with xi=0 is pure exploitation of the mean improvement.
    EI with xi > 0 requires a minimum improvement xi before counting it.
    So, high xi = more conservative/exploitative. Low xi = more aggressive/exploratory.
    
    Therefore:
    When f_best is bad (far from 0), we want low xi (aggressive).
    When f_best is good (close to 0), we want high xi (conservative).
    
    Let's map f_best to xi.
    If f_best > 0.1, xi = 0.0 (pure EI, very aggressive)
    If f_best < 0.01, xi = 0.05 (more conservative)
    
    We can use a smooth transition.
    Let threshold_low = 0.01
    Let threshold_high = 0.1
    If f_best >= threshold_high: xi = 0.0
    If f_best <= threshold_low: xi = 0.05
    Else: linear interpolation.
    
    Additionally, we can add a small UCB component that fades out as we get closer to the optimum.
    ucb_weight = max(0, (f_best - threshold_low) / (threshold_high - threshold_low)) * max_ucb_weight
    
    Let's try:
    score = EI(mu, sigma, f_best, xi) + ucb_weight * sigma
    
    This ensures that when we are far from the optimum, we have a strong exploration bonus (UCB term)
    and an aggressive EI (low xi). As we get closer, the UCB term vanishes and EI becomes more
    conservative (higher xi), focusing on refining the best point found.
    """
    start_time = time.time()
    
    # Adaptive parameters
    # Theoretical minimum is 0.
    # Weierstrass on [-0.5, 0.5] typically starts with values around 0.5-1.0.
    
    f_best_val = f_best if np.isfinite(f_best) else 1.0
    
    # Define the schedule
    # When f_best is "bad" (high), we want high exploration.
    # When f_best is "good" (low), we want low exploration.
    
    # xi for EI
    # Low xi -> more exploration (accepts smaller improvements)
    # High xi -> less exploration (requires significant improvement)
    
    # Let's interpolate xi from 0.0 (bad f_best) to 0.02 (good f_best)
    # Actually, 0.0 is the standard EI. 0.01-0.05 is common for exploitation.
    
    # Let's define a "confidence" score based on how good f_best is.
    # Confidence is high when f_best is close to 0.
    # Let's say confidence = 1 - min(f_best / 0.5, 1.0)
    # If f_best = 0.5, conf = 0. If f_best = 0, conf = 1.
    
    # Clip f_best to [0, 0.5] for the confidence calculation
    f_best_clipped = np.clip(f_best_val, 0.0, 0.5)
    confidence = 1.0 - (f_best_clipped / 0.5)
    
    # xi varies from 0.0 (low confidence) to 0.05 (high confidence)
    xi = 0.05 * confidence
    
    # UCB weight varies from 0.5 (low confidence) to 0.0 (high confidence)
    # This ensures we explore heavily at first.
    ucb_weight = 0.5 * (1.0 - confidence)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (f_best_val - mu - xi) / (sigma + _EPS)
        ei = (f_best_val - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    scores = ei + ucb_weight * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check: if we are taking too long, just return EI (fastest)
    if time.time() - start_time > 250:
         with np.errstate(divide='ignore', invalid='ignore'):
             z = (f_best_val - mu) / (sigma + _EPS)
             ei_fast = (f_best_val - mu) * norm.cdf(z) + sigma * norm.pdf(z)
         ei_fast = np.maximum(ei_fast, 0.0)
         return np.where(np.isfinite(ei_fast), ei_fast, 0.0)

    return scores
# EVOLVE-BLOCK-END
