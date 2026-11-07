"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track the number of evaluations per seed
# The problem states "For each seed... the evaluator starts", implying a fresh context.
# However, since we don't have a reset signal, we assume the function is called
# sequentially for a single run (seed). If the evaluator reuses the process for multiple seeds
# without resetting globals, this could be an issue. But typically in these benchmarks,
# the "seed" loop is the outer loop and the BO loop is the inner loop.
# To be safe, we can infer the step count from the number of calls if we assume
# the function is called exactly once per BO step.
# Total steps: 5 initial + 25 BO steps = 30 evaluations.
# The function is called 25 times (for the BO steps).
# We track the call count to determine the phase.

_call_count = 0
_START_TIME = time.time()
_DEADLINE = 330 - 10  # 10s safety margin

def acquisition(mu, sigma, f_best):
    global _call_count, _START_TIME, _DEADLINE
    
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a deterministic "exploit" score if we are running out of time
        # This ensures we don't crash, though the result might be suboptimal.
        # Actually, if we are past deadline, the evaluator kills us.
        # The guard is to prevent long computations *inside* this function if we add them.
        # For now, keep it fast.
        pass

    _call_count += 1
    
    # Total BO steps is 25.
    # Step 1-10: High exploration (Global)
    # Step 11-25: Exploitation (Local)
    
    n_total = 25
    current_step = min(_call_count, n_total)
    
    # Decay factor: starts high, decays exponentially
    # At step 1, beta should be large (e.g., 5.0)
    # At step 25, beta should be small (e.g., 0.5)
    # beta = beta_max * exp(-lambda * step)
    # Let's use a linear decay for simplicity and robustness, or a specific curve.
    # Linear: beta = 5.0 - (5.0 - 0.5) * (current_step - 1) / (n_total - 1)
    # This ensures a smooth transition.
    
    beta_start = 5.0
    beta_end = 0.5
    progress = (current_step - 1) / max(1, n_total - 1)
    beta = beta_start - (beta_start - beta_end) * progress
    
    # The directive mentions: "Apply a Gaussian kernel to the posterior standard deviation 
    # to create a spatially smoothed uncertainty map"
    # We don't have coordinates. We only have mu and sigma vectors.
    # Without coordinates, we cannot apply a spatial Gaussian kernel to the sigma values 
    # in a meaningful geometric way (i.e., smoothing neighbor values in space).
    # However, we can interpret "spatially smoothed" as a form of regularization 
    # or we can assume the GP sigma already captures the spatial correlation.
    # If we strictly follow the directive to apply a Gaussian kernel to sigma, 
    # and we lack coordinates, we might approximate by sorting or just using the raw sigma 
    # if we assume the GP length-scale (0.2) already provides the smoothing.
    # Given the constraint "No coordinates are passed", a true spatial smoothing is impossible 
    # unless we assume an ordering or ignore the spatial aspect and just use the magnitude.
    # Let's stick to the UCB with dynamic beta as the primary driver, which is the 
    # "structural change" from the threshold-based heuristic.
    
    # Ensure sigma is safe
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization: Score = -mu + beta * sigma
    # Higher score is better.
    # We want low mu (good) and high sigma (uncertain).
    scores = -mu + beta * sigma_safe
    
    # Optional: Blend with Expected Improvement (EI) for better local exploitation late in the game?
    # The directive specifically asks for UCB with dynamically adjusted beta.
    # Let's stick to pure UCB with the dynamic beta to keep the strategy distinct and simple.
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
