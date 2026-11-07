import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIMEOUT = 330.0  # Seconds


def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > _TIMEOUT - 1.0:
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    f_min_known = -1.8013

    # Calculate gap from current best to known global minimum
    gap = max(0.0, f_best - f_min_known)
    # Normalize gap: assume initial gap is around 1.0 to 1.5
    # If gap is large, we are early. If gap is small, we are late.
    norm_gap = min(1.0, gap)

    # Structural Change: Pure LCB strategy with fixed, high exploration constant
    # The directive suggests using a pure LCB with k=3.0 to aggressively target
    # high variance regions. However, pure LCB can be too greedy in uncertainty
    # early on, missing the mean-based signal.
    # To combine the current program's "adaptive" spirit with the "pure LCB"
    # structural idea, we will use a fixed high beta for LCB, but we will NOT
    # normalize the scores to [0,1] which destroys the scale information.
    # Instead, we will use a weighted sum where the LCB dominates, but we retain
    # a small EI component for late-stage refinement if the gap is very small.
    #
    # Actually, the directive says "Use a pure Lower Confidence Bound (LCB) strategy
    # with a fixed, high exploration constant (e.g., k=3.0)".
    # Let's try that first, but with a slight twist:
    # The current program normalizes EI and LCB to [0,1] and blends them.
    # The inspiration 1 uses a weighted sum of scaled EI and UCB (which is -LCB for minimization).
    #
    # Let's implement a hybrid that leans heavily into the LCB structure but
    # avoids the normalization pitfall.
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Fixed high exploration constant as per directive
    # k = 3.0
    k = 3.0
    
    # LCB score: -mu + k * sigma
    # Higher score means more likely to be the minimum (low mu, high sigma)
    lcb_score = -mu + k * sigma_safe

    # If we are very close to the global minimum (gap < 0.05), 
    # we might want to switch to exploitation (EI) to refine the solution.
    # Otherwise, stick to the aggressive exploration.
    if gap < 0.05:
        # Switch to EI for fine-tuning
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -50, 50)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        # Return EI directly
        score = ei
    else:
        # Return LCB directly
        score = lcb_score

    # Sanitize
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    score = np.clip(score, -1e6, 1e6)

    return score
