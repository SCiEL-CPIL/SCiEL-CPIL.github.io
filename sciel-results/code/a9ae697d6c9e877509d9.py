"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-scale coordinate-wise optimizer for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        # Fixed sequence of step sizes: Geometric decay from large to small
        # We use a deterministic set of scales. 
        # Lunacek has a "deep" minimum and a "shallow" minimum.
        # The deep minimum is usually far from the origin.
        # We will cycle through scales, always restarting from the best known point
        # if a full cycle doesn't improve significantly, or just continuing the descent.
        
        # Strategy:
        # 1. Generate a set of candidate starting points (deterministic grid + corners).
        # 2. For each candidate, perform a multi-scale coordinate descent.
        # 3. Keep global best.
        
        # To maximize early AOCC, we want to find *a* good point quickly.
        # Coordinate descent on Rastrigin-like functions can get stuck in local minima
        # if step size is too small. If too large, it oscillates.
        # The fixed geometric sequence handles this by starting large (to jump out of
        # shallow wells or cross valleys) and ending small (to fine-tune).
        
        # Scales: 5.0, 2.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001
        # This is about 12 scales.
        # In 5D, one full pass over all dimensions at all scales is 12 * 5 * 2 (pos/neg) = 120 calls.
        # We have 500 calls. So we can do ~4 full passes or multiple restarts.
        
        scales = [5.0, 2.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        
        # Initial points: 
        # 1. Origin
        # 2. Corners of [-5, 5]^5 (only 32 corners, but we can't test all with 500 budget if we also need descent)
        # 3. Deterministic pseudo-random points (Sobol-like or simple hash-based)
        
        # Let's use a small set of diverse initial points.
        # Point 0: [0, 0, 0, 0, 0]
        # Point 1: [4, 4, 4, 4, 4] (near corner, often where deep min is in Lunacek)
        # Point 2: [-4, -4, -4, -4, -4]
        # Point 3: [2, -2, 2, -2, 2]
        # Point 4: [-2, 2, -2, 2, -2]
        # Point 5: [5, 0, 0, 0, 0]
        # Point 6: [0, 5, 0, 0, 0]
        # Point 7: [0, 0, 5, 0, 0]
        # Point 8: [0, 0, 0, 5, 0]
        # Point 9: [0, 0, 0, 0, 5]
        
        init_points = [
            np.zeros(dim),
            np.array([4.0, 4.0, 4.0, 4.0, 4.0]),
            np.array([-4.0, -4.0, -4.0, -4.0, -4.0]),
            np.array([2.0, -2.0, 2.0, -2.0, 2.0]),
            np.array([-2.0, 2.0, -2.0, 2.0, -2.0]),
            np.array([5.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 5.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 5.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 5.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 5.0]),
            np.array([-5.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, -5.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -5.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, -5.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, -5.0]),
        ]
        
        # Function to perform deterministic coordinate descent from a start point
        def descend(x_start, remaining_budget):
            nonlocal calls_made, best_f, best_x
            x = x_start.copy()
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            if calls_made >= budget or time.time() > deadline:
                return x, fx

            # Iterate through scales
            # We stop early if a scale pass yields no improvement
            prev_f = fx
            for scale in scales:
                if calls_made >= budget or time.time() > deadline:
                    break
                
                improved_this_scale = False
                # For each dimension, try +step and -step
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    
                    # Try +
                    x_trial = x.copy()
                    x_trial[d] += scale
                    x_trial = np.clip(x_trial, -5, 5)
                    if not np.allclose(x_trial, x):
                        f_trial = func(x_trial)
                        calls_made += 1
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            improved_this_scale = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                    
                    if calls_made >= budget or time.time() > deadline:
                        break
                    
                    # Try -
                    x_trial = x.copy()
                    x_trial[d] -= scale
                    x_trial = np.clip(x_trial, -5, 5)
                    if not np.allclose(x_trial, x):
                        f_trial = func(x_trial)
                        calls_made += 1
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            improved_this_scale = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                
                # If no improvement in this scale, we might be in a local min or flat area.
                # We continue to smaller scales to ensure we squeeze out any remaining gradient.
                # However, if we have a lot of budget left and no improvement for 2 scales,
                # we could break, but for now we just proceed to smaller scales.
                
            return x, fx

        # Strategy:
        # We have 500 calls.
        # Descent from one point takes ~120-200 calls depending on early breaks.
        # We can afford 2-3 deep descents.
        # To maximize AOCC, we should try the most promising points first.
        # Points near corners (like [4,4,4,4,4]) are often closer to the global min in Lunacek.
        
        # Order of initial points to try:
        # 1. [4,4,4,4,4] (High probability of deep min)
        # 2. [-4,-4,-4,-4,-4] (Symmetric alternative)
        # 3. [0,0,0,0,0] (Center)
        # 4. [2,-2,2,-2,2]
        # 5. [-2,2,-2,2,-2]
        # ... and so on
        
        ordered_inits = [
            np.array([4.0, 4.0, 4.0, 4.0, 4.0]),
            np.array([-4.0, -4.0, -4.0, -4.0, -4.0]),
            np.zeros(dim),
            np.array([2.0, -2.0, 2.0, -2.0, 2.0]),
            np.array([-2.0, 2.0, -2.0, 2.0, -2.0]),
            np.array([5.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 5.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 5.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 5.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 5.0]),
            np.array([-5.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, -5.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -5.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, -5.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, -5.0]),
        ]
        
        # We will attempt to descend from each point.
        # If budget runs out, we stop.
        for init in ordered_inits:
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Estimate remaining budget
            # If we have < 50 calls left, a full descent might be too expensive to guarantee improvement
            # but partial descent is still better than nothing.
            # However, to be efficient, we only start a new descent if we have enough budget 
            # to at least do a few scales.
            if budget - calls_made < 20:
                break
                
            descend(init, budget - calls_made)
            
            # If we found a very good solution, we might stop early to save time?
            # No, AOCC rewards early progress, but we have fixed budget.
            # We just need to make sure we use the budget effectively.
            # If best_f is very low (e.g. < -50), further improvements are marginal for AOCC
            # (since log scale compresses large errors). But let's use all budget to be safe.
            
        # Final check: if we have budget left and best_x is not None,
        # we can do a final fine-tuning pass on best_x with very small steps
        # to ensure we are at the bottom of the local minimum we found.
        if best_x is not None and calls_made < budget:
            # Use a very small scale for final polish
            polish_scales = [0.001, 0.0005, 0.0002, 0.0001]
            x = best_x.copy()
            fx = best_f
            for scale in polish_scales:
                if calls_made >= budget or time.time() > deadline:
                    break
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    # Try +
                    x_trial = x.copy()
                    x_trial[d] += scale
                    x_trial = np.clip(x_trial, -5, 5)
                    if not np.allclose(x_trial, x):
                        f_trial = func(x_trial)
                        calls_made += 1
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                    if calls_made >= budget or time.time() > deadline:
                        break
                    # Try -
                    x_trial = x.copy()
                    x_trial[d] -= scale
                    x_trial = np.clip(x_trial, -5, 5)
                    if not np.allclose(x_trial, x):
                        f_trial = func(x_trial)
                        calls_made += 1
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
