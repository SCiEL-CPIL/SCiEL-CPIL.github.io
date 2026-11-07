import numpy as np
import time
import warnings
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)
        
        d = self.dim
        lb = np.full(d, -5.0)
        ub = np.full(d, 5.0)
        bounds = [(-5.0, 5.0)] * d

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
                
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, lb, ub)
            
            try:
                f = float(func(x))
                if not np.isfinite(f):
                    f = np.inf
            except Exception:
                f = np.inf
                
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diverse Initialization
        # Use Latin Hypercube Sampling for coverage and Axis-Aligned points for structure
        num_lhs = 40
        num_axis = 20
        candidates = []
        
        # Latin Hypercube Sampling
        # Simple LHS implementation
        for _ in range(num_lhs):
            if calls_made >= self.budget or time.time() > deadline:
                break
            # Generate LHS points
            # For simplicity and speed, we use random permutation of bins
            bins = np.linspace(-5.0, 5.0, num_lhs + 1)
            x = np.zeros(d)
            for j in range(d):
                # Pick a random bin index
                bin_idx = rng.integers(0, num_lhs)
                # Pick a random point within that bin
                x[j] = rng.uniform(bins[bin_idx], bins[bin_idx + 1])
            f = call_func(x)
            candidates.append((f, x))
            
        # Axis-Aligned Points
        # Points like (4, 0, 0, 0, 0), (-4, 0, 0, 0, 0), etc.
        # This helps with functions that have basins along axes (Rosenbrock, Rastrigin)
        axis_values = [4.0, -4.0, 2.0, -2.0, 0.0]
        for i in range(num_axis):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = np.zeros(d)
            # Pick a random axis and a random value
            axis = rng.integers(0, d)
            val = axis_values[rng.integers(0, len(axis_values))]
            x[axis] = val
            f = call_func(x)
            candidates.append((f, x))
            
        # Sort candidates by fitness
        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Hybrid Local Refinement
        # Select top candidates for local optimization
        num_local_runs = 12
        if len(candidates) < num_local_runs:
            num_local_runs = len(candidates)
            
        for i in range(num_local_runs):
            if calls_made >= self.budget or time.time() > deadline:
                break
                
            _, x_start = candidates[i]
            
            # Check if we have enough budget for local optimization
            remaining = self.budget - calls_made
            if remaining < 20:
                break
                
            # Try L-BFGS-B first (fast for smooth functions)
            x_lbfgs = x_start
            try:
                res = minimize(
                    call_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxfev': min(10, remaining // 2),
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    }
                )
                x_lbfgs = res.x
            except Exception:
                pass
            
            # Check if L-BFGS-B improved significantly
            # If not, or if we still have budget, try Nelder-Mead (robust for non-smooth)
            remaining = self.budget - calls_made
            if remaining > 10:
                try:
                    res_nm = minimize(
                        call_func,
                        x_lbfgs,
                        method='Nelder-Mead',
                        options={
                            'maxfev': min(15, remaining),
                            'xatol': 1e-8,
                            'fatol': 1e-8
                        }
                    )
                except Exception:
                    pass

        # Phase 3: Final Polish on Global Best
        if best_x is not None and calls_made < self.budget and time.time() < deadline:
            remaining = self.budget - calls_made
            if remaining > 5:
                try:
                    minimize(
                        call_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxfev': remaining,
                            'xatol': 1e-10,
                            'fatol': 1e-10
                        }
                    )
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(d)
            call_func(best_x)

        return best_x, best_f
