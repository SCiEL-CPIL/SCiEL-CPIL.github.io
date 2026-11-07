"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic grid search + adaptive (1+1)-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx

        # Phase 1: Deterministic Global Sampling
        # Cover the domain systematically to ensure early progress.
        # 1. Center
        # 2. 32 Vertices of [-5, 5]^5
        # 3. 10 Axis midpoints (±2.5 on each axis)
        
        points_to_eval = []
        
        # Center
        points_to_eval.append(np.zeros(n))
        
        # Vertices
        for mask in range(2 ** n):
            v = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(n)])
            points_to_eval.append(v)
            
        # Axis midpoints
        for i in range(n):
            for sign in [-1, 1]:
                mid = np.zeros(n)
                mid[i] = sign * 2.5
                points_to_eval.append(mid)
        
        # Evaluate these points
        for p in points_to_eval:
            if calls >= budget or time.time() > deadline - 10:
                break
            safe_call(p)
            
        # Fallback: if no points were evaluated (shouldn't happen), try origin
        if best_x is None:
            safe_call(np.zeros(n))

        # Phase 2: Adaptive (1+1)-ES Local Refinement
        # Start from the best point found in Phase 1.
        # Use an adaptive step size (sigma) that increases on failure and decreases on success.
        # This allows escaping shallow local minima and fine-tuning near the optimum.
        
        if best_x is not None and calls < budget and time.time() < deadline - 5:
            current_x = best_x.copy()
            current_f = best_f
            sigma = 1.0  # Initial step size
            
            # Parameters for (1+1)-ES adaptation
            # Standard 1/5th rule or similar:
            # If success: sigma *= 1.0 (or slight decrease)
            # If failure: sigma *= 1.5 (or similar increase)
            # We will use a simple adaptive scheme.
            
            max_iterations = budget - calls
            for _ in range(max_iterations):
                if calls >= budget or time.time() > deadline - 5:
                    break
                
                # Generate a random perturbation
                # Using a fixed seed for reproducibility within the run, but varying per instance?
                # The task says "hidden seeded shift", but our optimizer should be deterministic per instance?
                # "Deterministic grid search" was in the docstring. Let's use a fixed rng for the ES part
                # to be safe and reproducible.
                rng = np.random.default_rng(42)
                direction = rng.standard_normal(n)
                new_x = current_x + sigma * direction
                new_x = np.clip(new_x, -5.0, 5.0)
                
                # Avoid redundant calls if clipped to same point
                if np.allclose(new_x, current_x):
                    # Shrink sigma if we are stuck at a boundary
                    sigma *= 0.8
                    continue
                
                new_f = safe_call(new_x)
                
                if new_f < current_f:
                    # Success: accept new point, keep sigma or decrease slightly
                    current_x = new_x
                    current_f = new_f
                    sigma *= 0.9  # Slight contraction
                else:
                    # Failure: increase sigma to explore more
                    sigma *= 1.2
                    # Cap sigma to avoid huge jumps
                    if sigma > 5.0:
                        sigma = 5.0
                    # Floor sigma to avoid infinite loops with tiny steps
                    if sigma < 1e-6:
                        sigma = 1e-6

        return best_x, best_f
# EVOLVE-BLOCK-END
