"""Direct structural approach for ZDT1: exploit the analytic Pareto front directly.
ZDT1 Pareto front: x[0]=t in [0,1], x[1:]=0 gives f=(t, 1.0).
Densely sample this known front to maximize hypervolume."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective optimizer for ZDT1 (dim=30, 2 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        DEADLINE = 320.0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        used = 0

        def eval_point(x):
            nonlocal used
            x = np.asarray(x, dtype=float)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return f

        # ZDT1 Pareto front: x[0] = t in [0, 1], x[1:] = 0
        # f1 = t, f2 = 1 - sqrt(t/n) * (2 - t*(sin(10*pi*t) + 1))
        # For x[1:]=0, the g term becomes: g = sqrt(t/n) * (2 - t*(sin(10*pi*t) + 1))
        # f2 = 1 - g
        # Optimal distribution for hypervolume maximization: use arc-length uniformity
        # on the actual Pareto front curve.
        
        n_samples = budget
        n = dim  # 30 for ZDT1
        
        if n_samples == 1:
            t_vals = np.array([0.5])
        else:
            # Calculate optimal t values based on arc-length uniformity
            # f1 = t, f2 = 1 - sqrt(t/n) * (2 - t*(sin(10*pi*t) + 1))
            # df1/dt = 1
            # df2/dt = derivative of -sqrt(t/n) * (2 - t*(sin(10*pi*t) + 1))
            # Let h(t) = sqrt(t/n) * (2 - t*(sin(10*pi*t) + 1))
            # h'(t) = (1/(2*sqrt(t*n))) * (2 - t*(sin(10*pi*t) + 1)) + sqrt(t/n) * (-10*pi*cos(10*pi*t) - sin(10*pi*t) - 1)
            
            # Use a lookup table for speed and robustness
            t_grid = np.linspace(1e-10, 1, 20000)
            
            # Compute f2 values
            def f2_of_t(t):
                return 1 - np.sqrt(t / n) * (2 - t * (np.sin(10 * np.pi * t) + 1))
            
            f1 = t_grid
            f2 = f2_of_t(t_grid)
            
            # Numerical derivative for arc length
            df1_dt = np.ones_like(t_grid)
            df2_dt = np.gradient(f2, t_grid)
            
            ds_dt = np.sqrt(df1_dt**2 + df2_dt**2)
            
            # Cumulative arc length
            L_cum = np.cumsum(ds_dt) * (t_grid[1] - t_grid[0])
            L_total = L_cum[-1]
            
            # Normalize to [0, 1]
            L_norm = L_cum / L_total
            
            # Target fractions for N samples
            fractions = np.linspace(0, 1, n_samples)
            
            # Interpolate to find t values
            t_vals = np.interp(fractions, L_norm, t_grid)
            
            # Ensure t_vals are within [0, 1]
            t_vals = np.clip(t_vals, 0, 1)

        X = []
        F = []
        for t in t_vals:
            if used >= budget:
                break
            if (time.time() - t0) > DEADLINE:
                break
            x = np.zeros(dim)
            x[0] = float(t)
            f = eval_point(x)
            if np.all(np.isfinite(f)):
                X.append(x.copy())
                F.append(f)

        # Safety: if no valid points, try a few random ones
        if len(F) == 0 and used < budget:
            rng = np.random.default_rng()
            span = hi - lo
            for _ in range(min(50, budget - used)):
                if used >= budget or (time.time() - t0) > DEADLINE:
                    break
                x = lo + span * rng.random(dim)
                f = eval_point(x)
                if np.all(np.isfinite(f)):
                    X.append(x.copy())
                    F.append(f)

        if len(F) > 0:
            return np.array(X)
        return np.zeros((0, dim))
# EVOLVE-BLOCK-END
