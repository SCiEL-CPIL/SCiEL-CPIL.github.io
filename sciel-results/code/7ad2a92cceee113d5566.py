# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Deterministic multi-start local search with golden-section coordinate refinement for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        budget = self.budget

        state = {
            'calls': 0,
            'best_f': np.inf,
            'best_x': np.zeros(n)
        }

        def wrapped_func(x):
            if state['calls'] >= budget or time.time() > deadline:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            state['calls'] += 1
            if fx < state['best_f']:
                state['best_f'] = fx
                state['best_x'] = x_clipped.copy()
            return fx

        # Generate deterministic grid of starting points
        # Use a structured grid in [-4, 4]^5 with varying densities
        grid_points = []
        
        # Coarse grid: 3 points per dimension in [-4, 0, 4]
        coarse_vals = [-4.0, 0.0, 4.0]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            pt = np.array([coarse_vals[i], coarse_vals[j], coarse_vals[k], 
                                           coarse_vals[l], coarse_vals[m]])
                            grid_points.append(pt)
        
        # Medium grid: 5 points per dimension in [-4, -2, 0, 2, 4]
        medium_vals = [-4.0, -2.0, 0.0, 2.0, 4.0]
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    for l in range(5):
                        for m in range(5):
                            pt = np.array([medium_vals[i], medium_vals[j], medium_vals[k], 
                                           medium_vals[l], medium_vals[m]])
                            grid_points.append(pt)
        
        # Remove duplicates and limit to avoid too many calls
        seen = set()
        unique_points = []
        for pt in grid_points:
            key = tuple(np.round(pt, 6))
            if key not in seen:
                seen.add(key)
                unique_points.append(pt)
        
        # Shuffle deterministically
        rng = np.random.default_rng(42)
        rng.shuffle(unique_points)
        
        # Evaluate initial points
        max_initial = min(len(unique_points), 80)
        for idx in range(max_initial):
            if state['calls'] >= budget - 100 or time.time() > deadline:
                break
            pt = unique_points[idx]
            wrapped_func(pt)
        
        # Perform golden-section coordinate refinement on best point
        if state['calls'] < budget - 50:
            self._golden_section_refine(wrapped_func, state['best_x'].copy(), state, deadline)
        
        # Try refinement on second-best and third-best points if budget allows
        # Find top 3 distinct points from initial evaluation
        evaluated = []
        for pt in unique_points[:max_initial]:
            # Re-evaluate to get their values (they should be cached in state tracking, but we need to track them)
            # Actually, let's just track them during initial phase
            pass
        
        # For simplicity, just do one more refinement pass with smaller steps if budget allows
        if state['calls'] < budget - 20:
            self._golden_section_refine(wrapped_func, state['best_x'].copy(), state, deadline, initial_step=0.1)
        
        return state['best_x'], state['best_f']

    def _golden_section_refine(self, wrapped_func, start_x, state, deadline, initial_step=0.5):
        """Golden-section search along each coordinate sequentially, repeated until convergence or budget exhausted."""
        n = self.dim
        budget = self.budget
        
        x = start_x.copy()
        f_x = state['best_f']
        
        # Golden ratio
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        gr = 1.0 / phi  # ~0.618
        
        # Step sizes for each dimension, starting with initial_step
        steps = np.ones(n) * initial_step
        
        max_rounds = 10
        for round_num in range(max_rounds):
            if state['calls'] >= budget - 10 or time.time() > deadline:
                break
            
            improved = False
            for i in range(n):
                if state['calls'] >= budget - 10 or time.time() > deadline:
                    break
                
                # Perform golden-section search along dimension i
                x_opt, f_opt = self._golden_section_1d(wrapped_func, x, i, steps[i], state, deadline)
                
                if f_opt < f_x - 1e-10:
                    x = x_opt
                    f_x = f_opt
                    improved = True
                else:
                    # Shrink step size for this dimension
                    steps[i] *= 0.5
                    if steps[i] < 1e-5:
                        steps[i] = 1e-5
            
            if not improved:
                # If no improvement in a full round, shrink all steps and try again
                steps *= 0.5
                if np.all(steps < 1e-5):
                    break
            
            if f_x < state['best_f']:
                state['best_f'] = f_x
                state['best_x'] = x.copy()
        
        return x, f_x

    def _golden_section_1d(self, wrapped_func, x_base, dim_idx, step_size, state, deadline):
        """Perform 1D golden-section search along a single dimension."""
        n = self.dim
        x = x_base.copy()
        f_x = state['best_f']
        
        # Define search interval: [x[dim] - step, x[dim] + step]
        lo = x[dim_idx] - step_size
        hi = x[dim_idx] + step_size
        
        # Clip to bounds
        lo = max(lo, -5.0)
        hi = min(hi, 5.0)
        
        if hi - lo < 1e-8:
            return x, f_x
        
        # Golden-section points
        gr = 1.0 / ((1.0 + np.sqrt(5.0)) / 2.0)  # ~0.618
        
        x1 = hi - gr * (hi - lo)
        x2 = lo + gr * (hi - lo)
        
        # Evaluate at x1 and x2
        x_try1 = x.copy()
        x_try1[dim_idx] = x1
        x_try1 = np.clip(x_try1, -5, 5)
        f1 = wrapped_func(x_try1)
        
        if state['calls'] >= self.budget - 10 or time.time() > deadline:
            return x, f_x
        
        x_try2 = x.copy()
        x_try2[dim_idx] = x2
        x_try2 = np.clip(x_try2, -5, 5)
        f2 = wrapped_func(x_try2)
        
        if state['calls'] >= self.budget - 10 or time.time() > deadline:
            return x, f_x
        
        # Iteratively narrow the interval
        max_iter = 15
        for _ in range(max_iter):
            if state['calls'] >= self.budget - 10 or time.time() > deadline:
                break
            
            if hi - lo < 1e-7:
                break
            
            if f1 < f2:
                hi = x2
                x2 = x1
                f2 = f1
                x1 = hi - gr * (hi - lo)
            else:
                lo = x1
                x1 = x2
                f1 = f2
                x2 = lo + gr * (hi - lo)
            
            # Evaluate new point
            x_new = x.copy()
            x_new[dim_idx] = x1
            x_new = np.clip(x_new, -5, 5)
            f_new = wrapped_func(x_new)
            
            if state['calls'] >= self.budget - 10 or time.time() > deadline:
                break
            
            if f_new < f1:
                f1 = f_new
            else:
                # Keep f1 as the better of the two current points
                pass
        
        # Return the best point found
        # Check which of x1, x2, or original x is best
        candidates = [x_base.copy()]
        candidates.append(x_try1.copy())
        candidates.append(x_try2.copy())
        
        # Also check the midpoint of the final interval
        mid = (lo + hi) / 2.0
        x_mid = x_base.copy()
        x_mid[dim_idx] = mid
        x_mid = np.clip(x_mid, -5, 5)
        candidates.append(x_mid)
        
        best_x = x_base
        best_f = f_x
        
        for cand in candidates:
            if np.allclose(cand, best_x):
                continue
            f_cand = wrapped_func(cand)
            if state['calls'] >= self.budget - 10 or time.time() > deadline:
                break
            if f_cand < best_f:
                best_f = f_cand
                best_x = cand.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
