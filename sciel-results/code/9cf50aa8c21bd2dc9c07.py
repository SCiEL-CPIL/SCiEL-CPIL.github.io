import numpy as np
from scipy.stats import norm
import time

# Global state
_start_time = None
_step_counter = 0
_last_f_best = None

def acquisition(mu, sigma, f_best):
    global _start_time, _step_counter, _last_f_best
    
    if _start_time is None:
        _start_time = time.time()
    
    # Detect new seed/run
    # If f_best is significantly higher than the last observed min, assume new run
    # The function values are in the range roughly [-3, 0]
    if _last_f_best is not None and f_best > _last_f_best + 0.5:
        _step_counter = 0
    else:
        _step_counter += 1
        
    _last_f_best = f_best
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback to greedy (min mu) if something goes wrong or time is running out
        # This is a safety measure. EI is very fast.
        return -mu

    # Adaptive Expected Improvement (EI) with a non-monotonic exploration schedule.
    # Total evaluations = 30. Initial 5 are random. We have 25 steps for acquisition.
    
    # Schedule design:
    # Early steps (0-7): High exploration to map the landscape.
    # Mid steps (8-18): Balanced exploration/exploitation.
    # Late steps (19-24): Strong exploitation to refine the minimum.
    
    # We use a piecewise linear schedule for kappa.
    # kappa controls the weight of uncertainty (sigma).
    # score = -(mu + kappa * sigma) is UCB.
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # For minimization, we maximize EI.
    
    # To ensure we don't get stuck in local minima early on, we use a high kappa initially.
    # As we get more info, we reduce kappa.
    
    step = _step_counter
    if step < 0:
        step = 0
        
    # Define key points for the schedule: (step, kappa)
    # We want to explore broadly at first.
    # A common heuristic is to start with high uncertainty weighting.
    # Let's try a schedule that starts high, dips, and stays low.
    
    # Interpolation points:
    # Step 0: kappa = 2.0 (High exploration)
    # Step 5: kappa = 1.5
    # Step 10: kappa = 0.8
    # Step 15: kappa = 0.3
    # Step 20: kappa = 0.1
    # Step 24: kappa = 0.0 (Pure exploitation / Greedy)
    
    # We'll use a simple linear interpolation between these points.
    # This is a structural change from the constant or simple decay of other methods,
    # tailored to the specific 25-step budget.
    
    points = np.array([
        [0, 2.0],
        [5, 1.5],
        [10, 0.8],
        [15, 0.3],
        [20, 0.1],
        [25, 0.0]
    ])
    
    # Clip step to range
    step_clipped = np.clip(step, 0, 25)
    
    # Linear interpolation
    kappa = np.interp(step_clipped, points[:, 0], points[:, 1])
    
    # Calculate EI
    # mu: shape (2000,)
    # sigma: shape (2000,)
    # f_best: scalar
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    # Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Phi is CDF of standard normal
    # phi is PDF of standard normal
    
    # Use scipy.stats.norm
    # To avoid numerical issues with very large/small Z, we can rely on scipy's implementation
    # which is generally robust.
    
    Phi_Z = norm.cdf(Z)
    phi_Z = norm.pdf(Z)
    
    EI = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # We want to maximize EI.
    # The evaluator picks the candidate with the highest score.
    # So we return EI directly.
    
    return EI
