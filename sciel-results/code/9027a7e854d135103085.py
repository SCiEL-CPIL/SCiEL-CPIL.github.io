"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Structure-Exploiting Optimizer for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 330s kill

        call_count = 0
        max_calls = self.budget
        best_f = float('inf')
        best_x = np.zeros(self.dim)
        dim = self.dim

        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            call_count += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Structure-Exploiting Sampling
        # The Büche-Rastrigin function has local minima near integer coordinates.
        # We sample points where each coordinate is close to an integer in [-5, 5].
        integers = list(range(-5, 6))
        
        # Generate a diverse set of candidates based on integer grid
        # We will evaluate a subset of these to find promising basins.
        # To keep it fast, we don't evaluate all 11^5 = 161051 points.
        # We use random sampling of the integer grid with slight perturbations.
        
        n_initial_samples = 50
        candidates = []
        
        # Add the origin as a strong candidate
        candidates.append((counted_func(np.zeros(dim)), np.zeros(dim).copy()))
        
        for _ in range(n_initial_samples):
            if call_count >= max_calls or time.time() > deadline:
                break
            # Pick random integers for each dimension
            x_int = np.array([rng.choice(integers) for _ in range(dim)])
            # Add small noise to avoid exact singularities or to explore around the min
            x = x_int + rng.normal(0, 0.1, dim)
            f = counted_func(x)
            candidates.append((f, x.copy()))

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Coordinate Local Search with Ridge Jumping
        # We apply local search to the best candidates found in Phase 1.
        # The local search uses steps that can jump across the periodic ridges (step ~ 1.0).
        
        # We will attempt local search on the top 5 candidates
        top_k = min(5, len(candidates))
        
        for i in range(top_k):
            if call_count >= max_calls or time.time() > deadline:
                break
                
            f_init, x_init = candidates[i]
            x = x_init.copy()
            f_curr = f_init
            
            # Local search loop
            max_iter_local = 30
            for _ in range(max_iter_local):
                if call_count >= max_calls or time.time() > deadline:
                    break
                    
                improved = False
                # Try each dimension
                for d in range(dim):
                    if call_count >= max_calls or time.time() > deadline:
                        break
                    
                    # Try steps: 
                    # 1.0 and -1.0 (Jump to next/prev integer basin)
                    # 0.5, -0.5 (Half-step)
                    # 0.1, -0.1 (Fine tune)
                    steps = [1.0, -1.0, 0.5, -0.5, 0.1, -0.1]
                    
                    best_step_f = f_curr
                    best_step_x = x.copy()
                    
                    for step in steps:
                        x_trial = x.copy()
                        x_trial[d] += step
                        if x_trial[d] < -5 or x_trial[d] > 5:
                            continue
                        f_trial = counted_func(x_trial)
                        if f_trial < best_step_f:
                            best_step_f = f_trial
                            best_step_x = x_trial.copy()
                            improved = True
                    
                    if improved:
                        x = best_step_x
                        f_curr = best_step_f
                        if f_curr < best_f:
                            best_f = f_curr
                            best_x = x.copy()
                    
                    # If we found an improvement in this dimension, we might restart the dimension loop
                    # or just continue to the next dimension. 
                    # For speed, we just continue to next dimension.
                    
                if not improved:
                    # Try smaller steps if no improvement was found
                    small_improved = False
                    for d in range(dim):
                        if call_count >= max_calls or time.time() > deadline:
                            break
                        for step in [0.05, -0.05, 0.01, -0.01]:
                            x_trial = x.copy()
                            x_trial[d] += step
                            if x_trial[d] < -5 or x_trial[d] > 5:
                                continue
                            f_trial = counted_func(x_trial)
                            if f_trial < f_curr:
                                x = x_trial
                                f_curr = f_trial
                                if f_curr < best_f:
                                    best_f = f_curr
                                    best_x = x.copy()
                                small_improved = True
                                break
                        if small_improved:
                            break
                            
                    if not small_improved:
                        break # Stagnation for this candidate

        return best_x, best_f
# EVOLVE-BLOCK-END
