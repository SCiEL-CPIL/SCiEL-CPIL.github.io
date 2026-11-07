"""Optimized program for BBOB f21 (Gallagher 101 peaks) with Multi-Start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Two-phase optimizer:
    1. Global exploration via Latin Hypercube Sampling (20 points).
    2. Local exploitation via Multi-Start Nelder-Mead from the top 3 LHS points.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_f = float('inf')
        best_x = None
        
        def wrapped_func(x):
            nonlocal n_calls, best_f, best_x
            if n_calls >= budget:
                return float('inf')
            n_calls += 1
            try:
                val = func(x)
            except Exception:
                val = float('inf')
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)

        # --- Phase 1: Global Exploration via Latin Hypercube Sampling ---
        # Use 20 samples to cover the space well without wasting too much budget.
        n_lhs = 20
        if n_lhs >= budget:
            n_lhs = budget // 2
        
        # Generate LHS points in [0, 1]^dim
        lhs_indices = [rng.permutation(n_lhs) for _ in range(dim)]
        lhs_points = np.array([[i / n_lhs + rng.random() for i in idx] for idx in lhs_indices])
        
        # Map to [-5, 5]^dim
        lhs_points = lhs_points * 10.0 - 5.0
        
        # Evaluate LHS points
        for p in lhs_points:
            if n_calls >= budget or time.time() > deadline - 5:
                break
            x = np.clip(p, -5, 5)
            wrapped_func(x)
        
        # If we haven't made any valid call (should not happen), fallback to center
        if best_x is None:
            x = np.zeros(dim)
            wrapped_func(x)
            if best_x is None:
                return np.zeros(dim), 0.0

        # --- Phase 2: Multi-Start Nelder-Mead ---
        # Get the top 3 points from LHS based on their function values
        # We need to re-evaluate or store them. Let's store them.
        lhs_results = []
        # We already evaluated them, but we didn't store the mapping easily.
        # Let's just re-run the LHS evaluation logic but store results.
        # Actually, we can just use the best_x found so far and maybe a few others.
        # Better: re-generate LHS points and store their values.
        
        lhs_indices = [rng.permutation(n_lhs) for _ in range(dim)]
        lhs_points = np.array([[i / n_lhs + rng.random() for i in idx] for idx in lhs_indices])
        lhs_points = lhs_points * 10.0 - 5.0
        
        lhs_results = []
        for p in lhs_points:
            if n_calls >= budget or time.time() > deadline - 5:
                break
            x = np.clip(p, -5, 5)
            val = wrapped_func(x)
            lhs_results.append((val, x))
        
        # Sort by value
        lhs_results.sort(key=lambda x: x[0])
        
        # Select top 3 distinct starting points for Nelder-Mead
        start_points = []
        for val, x in lhs_results[:3]:
            if x is not None:
                start_points.append(x.copy())
        
        if not start_points:
            start_points = [best_x.copy()]

        # Run Nelder-Mead from each start point
        for x0 in start_points:
            if n_calls >= budget or time.time() > deadline - 5:
                break
            
            # Nelder-Mead parameters
            alpha = 1.0
            beta = 0.5
            gamma = 2.0
            delta = 0.5
            
            # Initialize simplex
            simplex = [x0.copy()]
            f0 = wrapped_func(x0)
            
            for i in range(dim):
                xi = x0.copy()
                xi[i] += delta
                simplex.append(xi)
            
            # Evaluate initial simplex
            f_values = []
            for i in range(dim + 1):
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                f = wrapped_func(simplex[i])
                f_values.append(f)
            
            # Nelder-Mead Loop
            max_iter = 200
            for _ in range(max_iter):
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                
                # Sort simplex by fitness
                idx = np.argsort(f_values)
                simplex = [simplex[i] for i in idx]
                f_values = [f_values[i] for i in idx]
                
                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5, 5)
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                f_r = wrapped_func(x_r)
                
                if f_values[0] <= f_r < f_values[-2]:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                elif f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if n_calls >= budget or time.time() > deadline - 5:
                        break
                    f_e = wrapped_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_values[-1]:
                        x_c = centroid + beta * (worst - centroid)
                    else:
                        x_c = centroid - beta * (worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    if n_calls >= budget or time.time() > deadline - 5:
                        break
                    f_c = wrapped_func(x_c)
                    
                    if f_c < f_values[-1]:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if n_calls >= budget or time.time() > deadline - 5:
                                break
                            f_values[i] = wrapped_func(simplex[i])
                        if n_calls >= budget or time.time() > deadline - 5:
                            break

        return best_x, best_f

# EVOLVE-BLOCK-END
