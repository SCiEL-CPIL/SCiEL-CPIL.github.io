"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track iteration count and time for dynamic behavior
_call_count = 0
_start_time = time.time()
_MAX_WALL_TIME = 300.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    global _call_count
    _call_count += 1
    
    # Deadline guard: if we are running out of time, just return a simple deterministic score
    # to ensure we finish within the time limit.
    elapsed = time.time() - _start_time
    if elapsed > _MAX_WALL_TIME:
        # Fallback: simple cost-aware score
        return (f_best - mu) / (cost + 1e-6)

    # Thompson Sampling Strategy
    # Draw a single sample from the posterior distribution for each candidate
    # f_s = mu + sigma * z, where z ~ N(0, 1)
    # We use a fixed seed for reproducibility within the run? 
    # The prompt says "seeded instances" but the acquisition function is called in a loop.
    # If we use a fixed seed, the random numbers will be the same every call, which might cause issues 
    # if the GP changes. It's better to use the global RNG or a seed derived from call count.
    # However, Thompson Sampling typically uses a fresh random draw each time to explore.
    # Let's use np.random.default_rng with a seed based on call count to ensure diversity but reproducibility if needed.
    # Actually, standard practice is just to use the global RNG or a dedicated one.
    # Since we want to break symmetry and explore, let's use a dedicated RNG seeded by call count.
    
    rng = np.random.default_rng(_call_count)
    z = rng.standard_normal(len(mu))
    
    # Sample from the posterior
    f_s = mu + sigma * z
    
    # Cost adjustment
    # The cost is 1 + mean(x_scaled). Range [1, 2].
    # We want to maximize the "value" per unit cost.
    # Since we are minimizing f, lower f_s is better.
    # We want to select the candidate with the lowest f_s, adjusted for cost.
    # Score = -f_s / (cost - 0.5)
    # Using (cost - 0.5) shifts the range to [0.5, 1.5], avoiding division by zero and making the relative difference more significant.
    # Alternatively, we can just use -f_s / cost.
    # Let's try -f_s / (cost + 0.1) to be safe, or stick to the directive's suggestion: f_s / (cost - 0.5).
    # The directive says: "selects the candidate maximizing f_s / (cost - 0.5)".
    # Wait, if we minimize f, we want to MINIMIZE f_s.
    # So we should select the candidate with the MINIMUM value of f_s / (cost - 0.5).
    # But the function returns scores where HIGHER is better.
    # So we should return - (f_s / (cost - 0.5)).
    
    # Let's verify the sign.
    # If f_s is low (good), -f_s is high (good score).
    # If cost is low (good), denominator is small, magnitude of -f_s/cost is larger.
    # So -f_s / (cost - 0.5) seems correct for higher=better.
    
    # Ensure cost - 0.5 is not zero or negative.
    # cost is in [1, 2], so cost - 0.5 is in [0.5, 1.5]. Safe.
    
    scores = -f_s / (cost - 0.5)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
