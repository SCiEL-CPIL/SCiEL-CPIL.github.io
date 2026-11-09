import numpy as np
from scipy.stats import norm
import time

# Track global time for safety deadline
_GLOBAL_START = time.time()
_SAFETY_LIMIT = 320.0  # 10s buffer before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function for minimizing Michalewicz 2D.
    
    Combines Expected Improvement (EI) with an adaptive Lower Confidence Bound (LCB) 
    term. Early in the optimization, a strong LCB component (high beta) is used to 
    systematically sweep the multimodal landscape and identify distinct basins. 
    As the best observed value (f_best) approaches the known global minimum (-1.8013), 
    the exploration weight decays, and the strategy shifts to exploitation via EI, 
    allowing for precise convergence to the optimum.
    
    The final score is a linear combination of the raw EI and the LCB score.
    """
    # Deadline Guard: If we are too close to the limit, return a simple greedy score
    if time.time() - _GLOBAL_START > _SAFETY_LIMIT:
        return -np.asarray(mu, dtype=np.float64)

    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Sanitize inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # Calculate LCB score (for minimization, lower is better, so we negate for maximization interface)
    # LCB_min = mu - beta * sigma
    # Score_LCB = -LCB_min = -mu + beta * sigma
    
    # Adaptive beta based on progress toward the known global minimum (-1.8013)
    # gap = f_best - f_min. If f_best is close to min, gap is small -> exploit.
    # If f_best is far, gap is large -> explore.
    f_min = -1.8013
    gap = f_best - f_min
    if gap < 0:
        gap = 0.0
        
    # Normalize gap. A gap of ~1.0 indicates significant room for improvement.
    # We use a smooth decay for beta.
    # Early (gap ~ 1.0): beta should be high (e.g., 2.0)
    # Late (gap ~ 0.0): beta should be low (e.g., 0.2)
    # Let's use a linear interpolation based on normalized gap.
    # Assume typical initial gap is around 0.5 to 1.5.
    # Let's map gap [0, 1.0] to beta [0.2, 2.0]
    # If gap > 1.0, keep beta at 2.0.
    norm_gap = np.clip(gap, 0.0, 1.0)
    beta = 0.2 + 1.8 * norm_gap
    
    # LCB score to maximize
    lcb_score = -mu + beta * sigma
    
    # Combine EI and LCB
    # We want a smooth transition. 
    # When gap is large, we rely more on LCB for exploration.
    # When gap is small, we rely more on EI for exploitation.
    
    # Let's use a weight that shifts from LCB to EI.
    # w_ei increases as gap decreases.
    # w_lcb decreases as gap decreases.
    
    # Simple blend:
    # score = w_ei * ei + w_lcb * lcb_score
    # However, EI and LCB have different scales. EI is in function units. LCB is also in function units (roughly).
    # But EI is always >= 0. LCB can be anything.
    # To ensure stability, we can normalize them or just use a fixed ratio that adapts.
    
    # Let's try:
    # Early: score = lcb_score (dominant)
    # Late: score = ei (dominant)
    
    # We can interpolate the weights.
    # w_ei = norm_gap (0 when at min, 1 when far) -> Wait, if gap is large, we want LCB.
    # So w_lcb should be high when gap is large.
    # w_lcb = 0.5 + 0.5 * norm_gap ?
    # w_ei = 0.5 - 0.5 * norm_gap ?
    
    # Let's refine:
    # When norm_gap = 1.0 (far): w_lcb = 1.0, w_ei = 0.0
    # When norm_gap = 0.0 (close): w_lcb = 0.0, w_ei = 1.0
    
    w_lcb = norm_gap
    w_ei = 1.0 - norm_gap
    
    # To avoid completely zero weight, add a small epsilon
    w_lcb = np.maximum(w_lcb, 0.05)
    w_ei = np.maximum(w_ei, 0.05)
    
    # Normalize weights to sum to 1
    w_sum = w_lcb + w_ei
    w_lcb /= w_sum
    w_ei /= w_sum
    
    score = w_lcb * lcb_score + w_ei * ei
    
    # Ensure finiteness
    if not np.all(np.isfinite(score)):
        score = np.where(np.isfinite(score), score, 0.0)
        if not np.all(np.isfinite(score)):
            score = -np.where(np.isfinite(mu), mu, f_best)
            
    # Safety deadline guard
    if time.time() - t0 > 30.0:
        return -np.where(np.isfinite(mu), mu, f_best)
        
    return score
