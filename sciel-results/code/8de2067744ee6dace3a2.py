"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start (1+1)-ES with stratified initial sampling and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls
            if calls >= self.budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            return fx

        # Phase 1: Stratified Random Sampling to find promising basins
        # We use 50 initial points to cover the space [-4, 4]^5
        num_initial = 50
        initial_points = []
        for i in range(num_initial):
            if calls >= self.budget or time.time() > deadline:
                break
            # Use stratified sampling: divide each dim into 5 bins?
            # Simpler: just random uniform, but ensure first few are spread out
            if i == 0:
                x = np.zeros(self.dim)
            elif i == 1:
                x = rng.uniform(-4, 4, self.dim)
            elif i == 2:
                x = rng.uniform(-4, 4, self.dim)
            else:
                x = rng.uniform(-4, 4, self.dim)
            
            fx = evaluate(x)
            if fx is None:
                break
            initial_points.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Sort initial points by fitness to process best basins first
        initial_points.sort(key=lambda item: item[0])
        
        # Phase 2: Multi-Start (1+1)-ES
        # We will try to optimize the top K initial points
        # K is determined by remaining budget
        top_k = 5  # Try top 5 basins
        
        sigma_min = 0.01
        sigma_max = 5.0
        stall_threshold = 30  # Steps without improvement
        max_local_steps = 80  # Max steps per local search attempt

        for idx in range(min(top_k, len(initial_points))):
            if calls >= self.budget or time.time() > deadline:
                break
            
            # Start from the idx-th best initial point
            f_start, x_start = initial_points[idx]
            x = x_start.copy()
            fx = f_start
            sigma = 1.0
            
            # (1+1)-ES
            stall_count = 0
            for step in range(max_local_steps):
                if calls >= self.budget or time.time() > deadline:
                    break
                
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.25, sigma_max)
                    stall_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.75, sigma_min)
                    stall_count += 1
                    if stall_count >= stall_threshold:
                        break
            
            # If we improved the global best, we might stop early if we are very close to optimum?
            # No, always try next basin if budget allows.
        
        # Phase 3: Final Polish with Nelder-Mead
        if best_x is not None and calls < self.budget and time.time() < deadline:
            try:
                eval_count = [0]
                def wrapped_func(x):
                    nonlocal calls
                    if calls >= self.budget:
                        return np.inf
                    x = np.clip(x, -5.0, 5.0)
                    fx = float(func(x))
                    calls += 1
                    eval_count[0] += 1
                    return fx

                remaining = self.budget - calls
                # Nelder-Mead can be expensive, limit iterations
                max_iter = min(50, max(10, remaining // 5))
                
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6,
                                        'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
