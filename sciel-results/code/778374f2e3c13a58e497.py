"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Nelder-Mead Optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin for 330s limit
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Coarse Sampling to find diverse starting points
        # We use a fraction of the budget for initial exploration
        # For dim=5, 300 calls is small. Let's use 50 for sampling, 250 for local search.
        n_starts = 10
        sample_budget = 50
        local_budget_total = budget - sample_budget
        
        candidates = []
        best_x = None
        best_f = np.inf
        
        # Generate initial candidates
        # Mix of random uniform and a few deterministic points (origin, corners)
        points = []
        # Origin
        points.append(np.zeros(dim))
        # Random uniform
        for _ in range(sample_budget - 1):
            p = rng.uniform(-5, 5, dim)
            points.append(p)
            
        # Evaluate initial points
        # We need to be careful with the budget.
        # We will evaluate all these, but if we run out of time, we stop.
        
        for p in points:
            if time.time() > deadline:
                break
            fp = func(p)
            if fp < best_f:
                best_f = fp
                best_x = p.copy()
            candidates.append((fp, p))
            
        if not candidates:
            # Fallback if even one call fails or takes too long
            return best_x if best_x is not None else np.zeros(dim), best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])
        
        # Select top N starts for Nelder-Mead
        # We want to spend remaining budget on these.
        # Nelder-Mead in 5D takes roughly (dim+1) * iterations to converge.
        # Let's allocate remaining budget equally among top starts.
        # However, early progress is key for AOCC. 
        # We should run the best start first to ensure we have a good best_f early.
        
        # Let's pick the top min(n_starts, len(candidates)) starts
        top_starts = [c[1] for c in candidates[:min(n_starts, len(candidates))]]
        
        # We have used len(points) calls so far.
        calls_made = len(points)
        remaining_budget = budget - calls_made
        
        if remaining_budget <= 0:
            return best_x, best_f
            
        # Allocate budget to each start
        # Give more budget to better starts? 
        # For AOCC, consistent improvement is good. 
        # Let's just split remaining budget among the top starts.
        # But we must ensure we don't exceed the time limit.
        
        starts_to_run = len(top_starts)
        budget_per_start = max(10, int(remaining_budget / starts_to_run))
        
        # Run Nelder-Mead on each start
        for i, start_x in enumerate(top_starts):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
                
            # Determine how many calls we can use for this start
            # We have calls_made so far.
            # We want to stop if we exceed budget or time.
            max_calls_this_start = min(budget_per_start, budget - calls_made)
            
            if max_calls_this_start < 10:
                break
                
            new_x, new_f, calls_used = self._nelder_mead(
                func, start_x.copy(), max_calls_this_start, dim, rng, deadline
            )
            
            calls_made += calls_used
            
            if new_f < best_f:
                best_f = new_f
                best_x = new_x.copy()
                
        return best_x, best_f

    def _nelder_mead(self, func, x0, max_calls, dim, rng, deadline):
        """
        Simple Nelder-Mead implementation.
        Returns best_x, best_f, calls_used.
        """
        # Initialize simplex
        simplex = [x0.copy()]
        f_vals = [func(x0)]
        calls_used = 1
        
        # Create other vertices
        for i in range(dim):
            xi = x0.copy()
            # Step size: 0.05 if x0[i] != 0 else 0.05
            step = 0.05 if abs(x0[i]) > 1e-8 else 0.05
            xi[i] += step
            # Clip to bounds
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            fi = func(xi)
            f_vals.append(fi)
            calls_used += 1
            
            if calls_used >= max_calls or time.time() > deadline:
                # Find best so far
                idx = np.argmin(f_vals)
                return simplex[idx], f_vals[idx], calls_used

        # Main loop
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        while calls_used < max_calls and time.time() < deadline:
            # Sort simplex by fitness
            order = np.argsort(f_vals)
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]
            
            x_worst = simplex[-1]
            f_worst = f_vals[-1]
            x_best = simplex[0]
            f_best = f_vals[0]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_used += 1
            
            if calls_used >= max_calls or time.time() > deadline:
                # Check if reflection is better than worst
                if f_r < f_worst:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                idx = np.argmin(f_vals)
                return simplex[idx], f_vals[idx], calls_used

            if f_r < f_best:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls_used += 1
                
                if calls_used >= max_calls or time.time() > deadline:
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    elif f_r < f_worst:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                    idx = np.argmin(f_vals)
                    return simplex[idx], f_vals[idx], calls_used

                if f_e < f_r:
                    simplex[-1] = x_e
                    f_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
            elif f_r < f_worst:
                # Accept reflection
                simplex[-1] = x_r
                f_vals[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (x_worst - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls_used += 1
                
                if calls_used >= max_calls or time.time() > deadline:
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    idx = np.argmin(f_vals)
                    return simplex[idx], f_vals[idx], calls_used

                if f_c < f_worst:
                    simplex[-1] = x_c
                    f_vals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_vals[i] = func(simplex[i])
                        calls_used += 1
                        if calls_used >= max_calls or time.time() > deadline:
                            idx = np.argmin(f_vals)
                            return simplex[idx], f_vals[idx], calls_used

        # Final best
        idx = np.argmin(f_vals)
        return simplex[idx], f_vals[idx], calls_used
# EVOLVE-BLOCK-END
