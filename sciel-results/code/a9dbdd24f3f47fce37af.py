"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start local optimization with L-BFGS-B for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        n_starts = 50
        calls_used = 0
        best_x = np.full(self.dim, np.nan)
        best_f = np.inf

        # Wrap func to count calls and enforce deadline
        def counted_func(x):
            nonlocal calls_used
            if calls_used >= self.budget:
                raise RuntimeError("Budget exceeded")
            elapsed = time.time() - start_time
            if elapsed > deadline:
                raise TimeoutError("Time limit exceeded")
            calls_used += 1
            return func(x)

        # Generate diverse starting points
        # Mix of random uniform, stratified, and lattice points
        starts = []
        
        # Stratified sampling: divide box into cells
        n_per_dim = 2  # 2^5 = 32 stratified points
        edges = np.linspace(-5, 5, n_per_dim + 1)
        for i in range(n_per_dim ** self.dim):
            idx = np.unravel_index(i, (n_per_dim,) * self.dim)
            point = []
            for d in range(self.dim):
                lo, hi = edges[d], edges[d + 1]
                point.append(rng.uniform(lo, hi))
            starts.append(np.array(point))
        
        # Add some random points
        while len(starts) < n_starts:
            starts.append(rng.uniform(-5, 5, self.dim))

        for s in starts:
            if calls_used >= self.budget:
                break
            elapsed = time.time() - start_time
            if elapsed > deadline:
                break

            try:
                # Use L-BFGS-B with numerical gradient
                # maxiter controls number of iterations, but each iteration
                # may use multiple function evaluations
                # We'll limit the number of iterations to keep within budget
                max_iter = max(1, (self.budget - calls_used) // 20)  # rough estimate
                
                result = minimize(
                    counted_func,
                    s,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
            except (RuntimeError, TimeoutError):
                break
            except Exception:
                continue

        # If we still have budget, do some fine-tuning around best_x
        if calls_used < self.budget and not np.isnan(best_f):
            elapsed = time.time() - start_time
            if elapsed < deadline:
                # Do a few more local refinements
                for _ in range(10):
                    if calls_used >= self.budget:
                        break
                    if time.time() - start_time > deadline:
                        break
                    try:
                        step = 0.5 * rng.standard_normal(self.dim)
                        x_new = np.clip(best_x + step, -5, 5)
                        f_new = counted_func(x_new)
                        if f_new < best_f:
                            best_f = f_new
                            best_x = x_new.copy()
                    except (RuntimeError, TimeoutError):
                        break
                    except Exception:
                        continue

        return best_x, best_f
# EVOLVE-BLOCK-END
