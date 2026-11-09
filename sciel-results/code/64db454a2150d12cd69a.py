"""Surrogate-assisted optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Derivative-free optimizer using Halton initialization and Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, use 320s to be safe
        deadline = start_time + 320.0
        
        dim = self.dim
        budget = self.budget
        
        # State tracking
        x_best = None
        f_best = np.inf
        eval_count = 0
        
        # Helper to check time and budget
        def check_deadline():
            return time.time() > deadline or eval_count >= budget

        # 1. Generate Halton sequence for global exploration
        n_halton = min(100, budget)
        halton_points = self._halton_sequence(n_halton, dim)
        # Scale to [-5, 5]
        halton_points = halton_points * 10.0 - 5.0
        halton_points = np.clip(halton_points, -5.0, 5.0)
        
        # 2. Phase 1: Global Exploration
        # Evaluate points until we find a good candidate or run out of time/budget
        # We will store the best 3 points to use as seeds for NM
        top_points = [] # List of (f, x)
        
        for i in range(n_halton):
            if check_deadline():
                break
            x = halton_points[i]
            f = func(x)
            eval_count += 1
            
            if f < f_best:
                f_best = f
                x_best = x.copy()
                
            # Maintain top 3
            top_points.append((f, x.copy()))
            top_points.sort(key=lambda item: item[0])
            if len(top_points) > 3:
                top_points.pop()
                
        if x_best is None:
            # Fallback
            x_best = np.zeros(dim)
            f_best = func(x_best)
            eval_count += 1
            if check_deadline():
                return x_best, f_best
            top_points = [(f_best, x_best.copy())]

        # 3. Phase 2: Local Exploitation using Nelder-Mead
        # Run NM from the best point
        if not check_deadline():
            x_best, f_best, eval_count = self._run_nm(func, x_best, f_best, eval_count, deadline)
            
        # Run NM from the 2nd best point if it's distinct enough and budget allows
        if len(top_points) > 1 and not check_deadline():
            f2, x2 = top_points[1]
            # Check if x2 is distinct from x_best
            if np.linalg.norm(x2 - x_best) > 0.1:
                x2, f2, eval_count = self._run_nm(func, x2, f2, eval_count, deadline)
                if f2 < f_best:
                    f_best = f2
                    x_best = x2.copy()
                    
        # Run NM from the 3rd best point
        if len(top_points) > 2 and not check_deadline():
            f3, x3 = top_points[2]
            if np.linalg.norm(x3 - x_best) > 0.1:
                x3, f3, eval_count = self._run_nm(func, x3, f3, eval_count, deadline)
                if f3 < f_best:
                    f_best = f3
                    x_best = x3.copy()

        # 4. Phase 3: Fine-tuning / Zoom-in
        # If we still have budget, do a local random search around the best point
        if not check_deadline():
            current_box_size = 1.0
            # Shrink if we improve, expand if we don't (up to a limit)
            for _ in range(50): # Max 50 rounds of zoom-in
                if check_deadline():
                    break
                
                improved = False
                # Generate a few random points in the box
                n_local = min(5, budget - eval_count)
                for _ in range(n_local):
                    if check_deadline():
                        break
                    x = x_best + (np.random.rand(dim) * 2.0 - 1.0) * current_box_size
                    x = np.clip(x, -5.0, 5.0)
                    f = func(x)
                    eval_count += 1
                    if f < f_best:
                        f_best = f
                        x_best = x.copy()
                        improved = True
                
                if improved:
                    current_box_size *= 0.5
                else:
                    current_box_size *= 1.2
                    if current_box_size > 5.0:
                        current_box_size = 5.0
                        
        return x_best, f_best

    def _halton_sequence(self, num_points, dim):
        """Generate a Halton sequence in [0, 1)^dim."""
        points = []
        for i in range(1, num_points + 1):
            point = []
            for d in range(1, dim + 1):
                f = 1.0
                r = 0.0
                i_temp = i
                while i_temp > 0:
                    i_temp, remainder = divmod(i_temp, d)
                    f /= d
                    r += f * remainder
                point.append(r)
            points.append(point)
        return np.array(points)

    def _run_nm(self, func, x0, f0, eval_count, deadline):
        """Run Nelder-Mead with strict evaluation counting."""
        dim = self.dim
        x_best = x0.copy()
        f_best = f0
        
        # We need to limit the number of function calls to stay within budget
        # and ensure we don't exceed the deadline.
        # Since we can't easily interrupt scipy.optimize.minimize, we set a maxiter
        # that is conservative.
        
        # Estimate max iterations based on remaining budget
        remaining_budget = self.budget - eval_count
        if remaining_budget <= 0:
            return x_best, f_best, eval_count
            
        # NM in d dimensions typically uses ~d+1 points per iteration.
        # Each iteration might use ~d+1 function calls.
        # Let's allow for ~ remaining_budget / (dim+1) iterations.
        max_iter = max(10, int(remaining_budget / (dim + 1)))
        
        count = [0]
        def wrapped_func(x):
            count[0] += 1
            return func(x)
            
        try:
            res = minimize(
                wrapped_func,
                x_best,
                method='Nelder-Mead',
                options={
                    'maxiter': max_iter,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            eval_count += count[0]
            
            if res.fun < f_best:
                f_best = res.fun
                x_best = res.x
        except Exception:
            pass
            
        # Check if we exceeded deadline during the call (unlikely for NM, but possible if func is slow)
        if time.time() > deadline:
            # We can't un-evaluate, so we just return what we have.
            pass
            
        return x_best, f_best, eval_count
# EVOLVE-BLOCK-END
