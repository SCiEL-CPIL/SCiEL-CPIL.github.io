import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for BBOB f21 (Gallagher 101 peaks).
    Strategy: Global exploration via Latin Hypercube Sampling, 
    followed by local refinement via L-BFGS-B on top candidates.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_val = np.inf
        self.best_x = None
        self.bounds = [(-5.0, 5.0)] * dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s is the kill limit. We stop at 315s to be safe.
        deadline = start_time + 315.0 

        # Wrapper to count calls and update global best
        def wrapped_func(x):
            if self.calls >= self.budget:
                return self.best_val
            if time.time() > deadline:
                return self.best_val
            
            # Clip to domain [-5, 5]
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            self.calls += 1
            
            if val < self.best_val:
                self.best_val = val
                self.best_x = x.copy()
                
            return val

        rng = np.random.default_rng(42)
        
        # Phase 1: Global Exploration using Latin Hypercube Sampling (LHS)
        # Use roughly 40-50% of the budget for global search to find good basins.
        n_global = int(self.budget * 0.4)
        
        def generate_lhs(n, dim, rng):
            # Generate n points in [0, 1]^dim using LHS
            points = np.zeros((n, dim))
            for i in range(dim):
                # Split [0, 1] into n intervals
                perm = rng.permutation(n)
                # Random offset within each interval
                offsets = rng.uniform(0, 1, n)
                points[:, i] = (perm + offsets) / n
            # Map to [-5, 5]
            points = points * 10.0 - 5.0
            return points

        lhs_points = generate_lhs(n_global, self.dim, rng)
        
        # Evaluate global points
        candidates = []
        for p in lhs_points:
            if self.calls >= self.budget or time.time() > deadline:
                break
            v = wrapped_func(p)
            candidates.append((v, p))
            
        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement using L-BFGS-B
        # Pick top K candidates to refine
        k_refine = min(5, len(candidates))
        
        # Reserve some budget for the final polish and safety
        # Each L-BFGS-B run might take a few dozen calls.
        # With budget 500 and 200 used for global, we have 300 left.
        # 5 runs * ~50 calls/run = 250 calls. Leaves 50 for polish.
        
        for i in range(k_refine):
            if self.calls >= self.budget - 10 or time.time() > deadline - 5:
                break
            
            init_val, init_x = candidates[i]
            
            # Define a callback to stop if we hit budget or time
            def stop_callback(xk):
                if self.calls >= self.budget - 5:
                    raise RuntimeError("Budget exceeded")
                if time.time() > deadline - 5:
                    raise RuntimeError("Time limit exceeded")
                
            try:
                # L-BFGS-B is efficient for low-dim smooth functions
                result = minimize(
                    wrapped_func,
                    init_x,
                    method='L-BFGS-B',
                    bounds=self.bounds,
                    options={
                        'maxiter': 100, 
                        'maxfun': self.budget - self.calls - 5,
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    },
                    callback=stop_callback
                )
                # The minimize function updates self.calls and self.best_val via wrapped_func
            except RuntimeError:
                # Stop condition met, proceed to next candidate or exit
                pass
            except Exception:
                # Ignore other optimization errors
                pass

        # Phase 3: Final Polish
        # If we have budget left, do a short L-BFGS-B run from the current best
        if self.calls < self.budget - 5 and time.time() < deadline - 5 and self.best_x is not None:
            try:
                def final_callback(xk):
                    if self.calls >= self.budget - 1:
                        raise RuntimeError("Budget exceeded")
                    if time.time() > deadline - 1:
                        raise RuntimeError("Time limit exceeded")

                minimize(
                    wrapped_func,
                    self.best_x,
                    method='L-BFGS-B',
                    bounds=self.bounds,
                    options={
                        'maxiter': 50,
                        'maxfun': self.budget - self.calls,
                        'ftol': 1e-15,
                        'gtol': 1e-10
                    },
                    callback=final_callback
                )
            except RuntimeError:
                pass
            except Exception:
                pass

        return self.best_x, self.best_val
