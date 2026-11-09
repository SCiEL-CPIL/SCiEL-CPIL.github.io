"""Optimized solver for BBOB f13 (Sharp Ridge) using Halton exploration and Powell's method."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Optimizer using Halton sequence for exploration and Powell's method for refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf
        box_low = -5.0
        box_high = 5.0

        def clip(x):
            return np.clip(x, box_low, box_high)

        def wrapped_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = clip(x)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def callback(xk):
            nonlocal calls_made
            if calls_made >= budget or time.time() > deadline:
                raise Exception("Budget or time limit exceeded")

        # Halton sequence for low-discrepancy initialization
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(index, dim, low, high):
            point = np.zeros(dim)
            for d in range(dim):
                base = d + 2  # 2, 3, 5, 7, 11, ...
                point[d] = low + (high - low) * halton(index + 1, base)
            return point

        # Phase 1: Broad exploration with Halton sequence
        n_explore = 40
        candidates = []
        
        for i in range(n_explore):
            if time.time() > deadline - 60:
                break
            x = halton_point(i, dim, box_low, box_high)
            f = wrapped_func(x)
            candidates.append((x, f))
            
        # Add structured points
        structured = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.array([1, -1, 1, -1, 1]) * 2.0,
        ]
        for x in structured:
            if time.time() > deadline - 60:
                break
            f = wrapped_func(x)
            candidates.append((x, f))

        # Sort candidates by function value
        candidates.sort(key=lambda item: item[1])
        
        # Phase 2: Local refinement using Powell's Method
        # Powell's method is effective for sharp ridges as it adapts search directions
        top_k = 3
        for i in range(top_k):
            if calls_made >= budget or time.time() > deadline - 30:
                break
            
            x0 = candidates[i][0]
            
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='Powell', 
                    options={'maxiter': 15, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False},
                    callback=callback
                )
                # Update best if this run found something better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Final Polish with fine-grained coordinate descent
        if calls_made < budget and time.time() < deadline - 5 and best_x is not None:
            current_x = best_x.copy()
            current_f = best_f
            step = 1e-4
            
            # Iterate a few times with decreasing step size
            for _ in range(5):
                if calls_made >= budget or time.time() > deadline - 5:
                    break
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline - 5:
                        break
                    
                    # Try positive step
                    trial = current_x.copy()
                    trial[d] += step
                    trial = clip(trial)
                    f_trial = wrapped_func(trial)
                    if f_trial < current_f:
                        current_x = trial
                        current_f = f_trial

                    # Try negative step
                    if calls_made >= budget or time.time() > deadline - 5:
                        break
                    trial = current_x.copy()
                    trial[d] -= step
                    trial = clip(trial)
                    f_trial = wrapped_func(trial)
                    if f_trial < current_f:
                        current_x = trial
                        current_f = f_trial
                
                # Decrease step size for next iteration
                step *= 0.5

            # Update global best if polish improved it
            if current_f < best_f:
                best_f = current_f
                best_x = current_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
