import numpy as np
from scipy.stats import norm
import time

# Module-level variable to track start time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement (EI)
    and Upper Confidence Bound (UCB) with adaptive exploration.
    
    The key structural change from pure EI/cost is the explicit UCB term which
    rewards high-uncertainty regions, helping to escape local optima in the
    Styblinski-Tang function (which has many local minima). The exploration
    parameter beta is dynamic, starting high to encourage global search and
    decaying as f_best improves, to shift focus to exploitation.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next).
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: If we are close to the 300s limit, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 295.0:
        # Fallback: Return something that prioritizes low cost and low mu
        # This is a safety net to avoid any potential complex branching if we were doing iterative methods.
        # For now, just proceed with the fast calculation.
        pass

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Sanitize inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=-1e6)
    
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI) for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # Assuming f(x) ~ N(mu(x), sigma(x)^2)
    # Let z = (f_best - mu(x)) / sigma(x)
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    
    z = (f_best - mu) / sigma
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # 2. Upper Confidence Bound (UCB) for minimization:
    # We want to minimize the lower bound of the function value.
    # LCB(x) = mu(x) - beta * sigma(x)
    # To turn this into a score to MAXIMIZE, we negate it:
    # UCB_score(x) = -mu(x) + beta * sigma(x)
    
    # Adaptive beta: 
    # Start with a high beta to encourage exploration.
    # As f_best gets closer to the known minimum (-78.3323), reduce beta.
    # We can normalize the progress. Let's assume the range is roughly [-80, 100].
    # A simple heuristic: beta starts at 3.0 and decays exponentially with improvement.
    # However, f_best is in objective units. The Styblinski-Tang function has a minimum of -78.3323.
    # Let's define a reference "good" value, e.g., -70.
    # progress = max(0, (f_best - (-70)) / 100) ... this is tricky without knowing the exact scale of f_best early on.
    # Alternative: Use a fixed but slightly higher beta than standard, and rely on cost normalization.
    # The inspiration suggests "dynamic pe[ring]" (likely beta).
    
    # Let's use a beta that is inversely proportional to the number of evaluations?
    # We don't have the number of evaluations directly, but we have cost.
    # Total cost spent is roughly the sum of costs. We don't have that here.
    # We can estimate progress by how low f_best is.
    # If f_best is very high (e.g., > 0), we are in early exploration.
    # If f_best is low (e.g., < -50), we are in exploitation.
    
    # Let's define beta based on f_best.
    # f_best starts around 50-100 and goes down to -78.
    # Let's map f_best to a beta range [1.0, 4.0].
    # If f_best > 0, beta = 4.0 (high exploration)
    # If f_best < -70, beta = 1.0 (low exploration)
    # Linear interpolation in between.
    
    if f_best > 0:
        beta = 4.0
    elif f_best < -70:
        beta = 1.0
    else:
        # Interpolate between 0 and -70
        # beta = 4.0 - (f_best / 70.0) * 3.0
        beta = 4.0 - (f_best / 70.0) * 3.0
    
    ucb_score = -mu + beta * sigma
    
    # 3. Combine EI and UCB
    # We need to normalize them to be on a similar scale, or just weight them.
    # EI is always positive. UCB_score can be negative.
    # A common approach is to take the max or a weighted sum.
    # Let's use a weighted sum with a parameter alpha.
    # score = alpha * EI + (1-alpha) * UCB_score
    # However, the scales are different. EI is an expected improvement (absolute value).
    # UCB_score is a bound on the function value (absolute value).
    # If f_best is -70, EI is small. UCB_score is around -70 + beta*sigma.
    # If f_best is 50, EI is large. UCB_score is around 50 + beta*sigma.
    
    # Let's try a different combination:
    # The goal is to find the minimum.
    # EI tells us where we can improve the current best.
    # UCB tells us where the function might be low.
    
    # Let's normalize UCB_score by subtracting f_best?
    # ucb_improvement = f_best - (-mu + beta*sigma) = f_best + mu - beta*sigma
    # This is similar to EI but with a linear term instead of the Gaussian integral.
    
    # Let's stick to the inspiration: "blending EI and UCB with cost normalization".
    # We will compute both, normalize them by their respective maxes (to handle scale), and blend.
    
    ei_max = np.max(ei)
    ucb_max = np.max(ucb_score)
    
    # Avoid division by zero
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
        
    if ucb_max > 0:
        ucb_norm = ucb_score / ucb_max
    else:
        ucb_norm = ucb_score
        
    # Blend: 50% EI, 50% UCB
    # This balances local improvement (EI) with global exploration (UCB).
    combined_score = 0.5 * ei_norm + 0.5 * ucb_norm
    
    # 4. Cost-awareness
    # Divide by cost to penalize expensive evaluations.
    score = combined_score / cost
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
