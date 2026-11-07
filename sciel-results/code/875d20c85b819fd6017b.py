# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer for WFG7 using Tchebycheff scalarization
    with Nelder-Mead local search to handle non-convex Pareto fronts.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj

        X_all = []
        F_all = []
        eval_count = [0]

        def evaluate(x):
            """Evaluate a point, record it, and return objectives."""
            if time.time() > deadline:
                return None
            if eval_count[0] >= self.budget:
                return None
            
            x = np.asarray(x, dtype=float).flatten()
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float).flatten()
            X_all.append(x.copy())
            F_all.append(f.copy())
            eval_count[0] += 1
            return f

        def time_ok():
            return time.time() < deadline and eval_count[0] < self.budget

        # Phase 1: Initial random sampling
        n_init = min(20, self.budget // 5)
        for i in range(n_init):
            if not time_ok():
                break
            x = lo + (hi - lo) * np.random.random(n)
            evaluate(x)

        if not F_all:
            return np.zeros((1, n))

        # Estimate ideal point z* (minimum of each objective found so far)
        F_arr = np.array(F_all)
        z_star = np.min(F_arr, axis=0)

        # Generate weight vectors for Tchebycheff scalarization
        # We want to cover the simplex uniformly.
        # For 3 objectives, we can use a grid on the 2D simplex.
        n_weights = 12
        weights = []
        
        # Add extreme points
        for i in range(m):
            w = np.zeros(m)
            w[i] = 1.0
            weights.append(w)
        
        # Add uniform random points on the simplex
        # Use Dirichlet distribution with alpha=1 for uniform distribution on simplex
        remaining = n_weights - m
        for _ in range(remaining):
            w = np.random.dirichlet(np.ones(m))
            weights.append(w)
            
        np.random.shuffle(weights)

        # Phase 2: Tchebycheff scalarization with Nelder-Mead
        # Tchebycheff function: max_i (lambda_i * |f_i(x) - z*_i|)
        # We minimize this.
        
        # We need to be careful with the budget per optimization.
        # Nelder-Mead can be expensive. We limit maxfev.
        maxfev_per_opt = 15
        maxiter_per_opt = 10
        
        for w in weights:
            if not time_ok():
                break
            
            # Define scalarized objective
            def tchebycheff_obj(x, w=w):
                x = np.clip(x, lo, hi)
                f = np.asarray(problem(x), dtype=float).flatten()
                # Calculate Tchebycheff value
                # We use absolute difference from z_star
                diff = np.abs(f - z_star)
                val = np.max(w * diff)
                return val
            
            # Note: tchebycheff_obj does NOT count evaluations in eval_count!
            # We must ensure that the optimizer's internal calls are tracked.
            # However, scipy.optimize.minimize calls the function directly.
            # We need a wrapper that counts.
            
            # Let's create a counting wrapper for the optimizer
            opt_count = [0]
            def counting_obj(x, w=w):
                if not time_ok():
                    return 1e10
                if opt_count[0] >= maxfev_per_opt:
                    return 1e10
                
                x = np.clip(x, lo, hi)
                f = np.asarray(problem(x), dtype=float).flatten()
                diff = np.abs(f - z_star)
                val = np.max(w * diff)
                
                # Record this evaluation in the global list
                # This is crucial: every call to problem must be recorded
                X_all.append(x.copy())
                F_all.append(f.copy())
                eval_count[0] += 1
                opt_count[0] += 1
                
                return val

            # Start from a random point
            x0 = lo + (hi - lo) * np.random.random(n)
            
            try:
                result = minimize(
                    counting_obj,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': maxiter_per_opt, 'maxfev': maxfev_per_opt, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                
                # The result.x is the best found by Nelder-Mead.
                # It has already been recorded in counting_obj.
                # We don't need to evaluate it again unless we want to be sure.
                # But counting_obj already recorded it.
                
            except Exception:
                continue

        # Phase 3: If budget remains, do some random sampling to fill gaps
        while time_ok() and eval_count[0] < self.budget:
            x = lo + (hi - lo) * np.random.random(n)
            evaluate(x)

        return np.array(X_all) if X_all else np.zeros((1, n))
# EVOLVE-BLOCK-END
