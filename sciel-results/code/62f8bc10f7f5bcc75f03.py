import numpy as np
import math
import time

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 110.0  # 110s safety margin before 120s limit

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    # Strategy: Use a high-resolution binomial tree (1200 steps) to compute
    # optimal exercise boundaries via backward induction.
    # This provides a much more accurate approximation of the optimal policy
    # than the 12-step tree, especially for the continuation value.
    
    n_steps = 1200
    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    df = math.exp(-r * dt)
    
    # Exercise dates: steps 100, 200, 300, ..., 1200
    # We only need to record boundaries at these steps.
    exercise_steps = {i * n_steps // n_exercises: i for i in range(1, n_exercises + 1)}
    
    # Grid for S values
    # Ensure the grid covers the possible range of S.
    # S_min = 100 * d^n_steps, S_max = 100 * u^n_steps
    # For n_steps=1200, range is approx [57, 175]. 
    # Let's use a wide grid to be safe, e.g., [0.1, 300].
    S_min = 0.1
    S_max = 300.0
    n_grid = 5000  # Higher resolution for better boundary accuracy
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # V at maturity (step n_steps)
    # At the last step, we must exercise if in the money.
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Boundaries: boundaries[j] for j=1..12 (j=1 is first exercise, j=12 is maturity)
    boundaries = np.zeros(n_exercises + 1)
    
    # Backward induction from step n_steps-1 down to step 1
    # We iterate backwards through all steps.
    # At each step, we check if it's an exercise date.
    
    # Optimization: We can vectorize the interpolation and update.
    # However, Python loop over 1200 steps is fast enough.
    
    for step in range(n_steps - 1, 0, -1):
        # Intrinsic value
        intrinsic = np.maximum(K - S_grid, 0.0)
        
        # Continuation value: df * [p * V(S*u) + (1-p) * V(S*d)]
        Su = S_grid * u
        Sd = S_grid * d
        
        # Interpolate V_next at Su and Sd
        # np.interp is fast but O(N log N) or O(N) if sorted. 
        # Since S_grid is sorted and Su/Sd are monotonic, we can use searchsorted for speed
        # or just rely on np.interp which is optimized.
        V_up = np.interp(Su, S_grid, V_next)
        V_down = np.interp(Sd, S_grid, V_next)
        
        continuation = df * (p * V_up + (1 - p) * V_down)
        
        # Value is max(intrinsic, continuation)
        V_current = np.maximum(intrinsic, continuation)
        
        # Check if this step is an exercise date
        if step in exercise_steps:
            idx = exercise_steps[step]
            # Find boundary: where intrinsic = continuation
            # For a put, we exercise when S is small.
            diff = intrinsic - continuation
            
            # Find the last index where diff >= 0
            exercise_mask = diff >= 0
            if np.any(exercise_mask):
                idxs = np.where(exercise_mask)[0]
                last_idx = idxs[-1]
                if last_idx < n_grid - 1:
                    diff0 = diff[last_idx]
                    diff1 = diff[last_idx + 1]
                    if diff1 != diff0:
                        t = diff0 / (diff0 - diff1)
                        boundaries[idx] = S_grid[last_idx] + t * (S_grid[last_idx + 1] - S_grid[last_idx])
                    else:
                        boundaries[idx] = S_grid[last_idx]
                else:
                    boundaries[idx] = S_grid[-1]
            else:
                boundaries[idx] = 0.0
        
        V_next = V_current
        
        # Check time budget periodically
        if step % 200 == 0:
            if time.time() > deadline - 5:
                # If we run out of time, we have partial boundaries.
                # Fill remaining with a heuristic or 0.
                # But 1200 steps is very fast, this shouldn't happen.
                pass

    # At maturity (j=12), we always exercise if S < K (since it's the last date)
    # The loop above sets boundaries[12] when step == 1200? 
    # Wait, the loop is range(n_steps - 1, 0, -1). 
    # So it goes down to 1. 
    # Step 1200 is the maturity step. 
    # My loop starts at 1199. 
    # I need to handle step 1200 (maturity) separately or adjust loop.
    # At step 1200, V is just intrinsic. Boundary is K.
    # My loop calculates V_current for step 1199 using V_next (which is V at 1200).
    # So boundaries[12] should be K.
    boundaries[12] = K

    class ExercisePolicy:
        def __init__(self, boundaries, K):
            self.boundaries = boundaries
            self.K = K
        
        def exercise(self, current_state, time):
            if isinstance(time, (int, float, np.integer, np.floating)):
                idx = int(time)
                idx = max(1, min(12, idx))
                b = self.boundaries[idx]
                return np.array(current_state) <= b
            else:
                times = np.array(time)
                times_clipped = np.clip(times, 1, 12).astype(int)
                b = np.array([self.boundaries[t] for t in times_clipped])
                return np.array(current_state) <= b
    
    return ExercisePolicy(boundaries, K)
