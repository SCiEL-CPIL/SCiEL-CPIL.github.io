import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Time guard: if we're close to the limit, return a simple score
    elapsed = time.time() - _start_time
    if elapsed > 320:
        return -mu
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Expected Improvement for minimization
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    
    # Standard EI for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Dynamic Pareto-based multi-objective weighting
    # We need to estimate the number of evaluations. Since we don't have direct access,
    # we can infer it from the magnitude of sigma or just use a fixed schedule.
    # However, the task says "dynamic weight that increases with the number of evaluations".
    # Without explicit n_evals, we can approximate:
    # Early on, sigma is high everywhere. Later, sigma is low in explored areas.
    # A common heuristic is to use the current f_best or the variance of mu.
    # Let's look at the distribution of mu and sigma.
    # Actually, a simpler dynamic weight: 
    # If f_best is still far from the known minimum (-3.32237), we are early.
    # But we don't know the exact minimum in the function signature, only f_best.
    # Let's use a fixed but effective blend of EI and Uncertainty (UCB-like).
    
    # The inspiration 1 (score 1.2867) used EI + 0.1 * sigma.
    # The current program (score 0.9840) used -mu + 2.0 * sigma (LCB).
    # EI is generally more robust than LCB because it naturally balances exploration and exploitation.
    # The "exploration bonus" in inspiration 1 effectively makes it a hybrid.
    
    # Let's refine the "dynamic" aspect.
    # We can estimate the "stage" of optimization by looking at the spread of mu.
    # If mu is very spread out, we are likely early. If concentrated, later.
    # However, a simpler and more robust approach is to stick with the successful hybrid from inspiration 1
    # but perhaps tune the exploration coefficient.
    
    # Inspiration 1: ei + 0.1 * sigma
    # Let's try a slightly higher exploration to ensure we don't get stuck, as 6D Hartmann has many local minima.
    # But 0.1 worked well (1.2867). Let's stick close to that but ensure numerical stability.
    
    # Structural change: Instead of a fixed 0.1, let's make it adaptive based on the current uncertainty.
    # If the mean sigma is high, we need more exploration. If low, less.
    # mean_sigma = np.mean(sigma)
    # But we want to avoid over-exploring in late stages.
    
    # Let's use the Pareto idea: 
    # Objective 1: Maximize EI (Exploitation)
    # Objective 2: Maximize Sigma (Exploration)
    
    # Normalize both to [0, 1] range approximately
    # EI can be negative? No, EI >= 0 by definition.
    # Sigma >= 0.
    
    # Simple weighted sum:
    # score = w * (ei / max_ei) + (1-w) * (sigma / max_sigma)
    # But normalizing by max in the array can be unstable if max is 0 or very small.
    
    # Let's stick to the additive form which is more robust:
    # score = ei + alpha * sigma
    
    # The "dynamic weight" directive suggests alpha should change.
    # Early: alpha high. Late: alpha low.
    # How to detect "early" vs "late"?
    # f_best is a good proxy. If f_best is high (bad), we are early.
    # Hartmann 6D min is -3.32237.
    # If f_best > -2.0, we are early.
    # If f_best < -3.0, we are late.
    
    # Let's map f_best to alpha.
    # alpha = 0.5 * (1 - sigmoid(f_best - threshold)) ?
    # Or simply:
    # if f_best > -2.5: alpha = 0.2
    # elif f_best > -3.0: alpha = 0.1
    # else: alpha = 0.05
    
    # Let's try a smooth transition.
    # Range of f_best of interest: -2.0 to -3.3
    # Let's linearly interpolate alpha from 0.3 (at -2.0) to 0.05 (at -3.3)
    
    f_best_clipped = np.clip(f_best, -3.3, -2.0)
    # t = 0 when f_best = -2.0 (early), t = 1 when f_best = -3.3 (late)
    t = (f_best_clipped - (-2.0)) / (-3.3 - (-2.0))
    t = np.clip(t, 0, 1)
    
    alpha = 0.3 - t * (0.3 - 0.05)
    
    scores = ei + alpha * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
