"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer for BBOB-style f19."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Track function calls to enforce budget
        call_count = 0

        def counted_func(x):
            nonlocal call_count
            call_count += 1
            return func(x)

        # Generate diverse starting points
        # 1. Origin
        # 2. Stratified random points (LHS-like)
        # 3. Points near boundaries
        n_starts = 8
        starts = []
        starts.append(np.zeros(n))
        
        # Add stratified points
        for i in range(n_starts - 1):
            # Stratified sampling in each dimension
            x = np.array([(i + rng.random()) / n_starts * 10 - 5 for _ in range(n)])
            starts.append(x)
            
        # Add a few random points for robustness
        for _ in range(4):
            starts.append(rng.uniform(-5, 5, size=n))

        best_x = None
        best_f = np.inf
        bounds = [(-5, 5)] * n

        for x0 in starts:
            if time.time() > deadline:
                break
            if call_count >= budget:
                break

            # Limit evaluations per run to allow multiple restarts
            # L-BFGS-B needs ~50-80 evals for good convergence in 5D
            remaining = budget - call_count
            max_evals_per_run = min(remaining, 60)
            if max_evals_per_run < 10:
                break

            try:
                res = minimize(counted_func, x0, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 100, 'maxfun': max_evals_per_run})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # If we still have budget, do a final polish on the best point
        if best_x is not None and call_count < budget and time.time() < deadline:
            remaining = budget - call_count
            if remaining > 10:
                try:
                    res = minimize(counted_func, best_x, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 100, 'maxfun': remaining})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Final coordinate-wise golden-section search for high-precision polish
        if best_x is not None and call_count < budget and time.time() < deadline:
            x = best_x.copy()
            f_curr = best_f
            
            # Golden section search on each coordinate
            phi = (1 + np.sqrt(5)) / 2  # golden ratio
            inv_phi = 1 / phi
            inv_phi2 = 1 / phi**2
            
            # Adaptive step size for each dimension
            for dim_i in range(n):
                if call_count >= budget or time.time() > deadline:
                    break
                
                # Initial step size based on current position
                step = 0.5
                # Try to find a good initial bracket
                x_minus = x.copy()
                x_plus = x.copy()
                x_minus[dim_i] = max(-5, x[dim_i] - step)
                x_plus[dim_i] = min(5, x[dim_i] + step)
                
                f_minus = counted_func(x_minus)
                f_plus = counted_func(x_plus)
                
                # If both are worse, try smaller steps
                if f_minus > f_curr and f_plus > f_curr:
                    # Try half step
                    step *= 0.5
                    x_minus[dim_i] = max(-5, x[dim_i] - step)
                    x_plus[dim_i] = min(5, x[dim_i] + step)
                    f_minus = counted_func(x_minus)
                    f_plus = counted_func(x_plus)
                    
                    if f_minus > f_curr and f_plus > f_curr:
                        # Try quarter step
                        step *= 0.5
                        x_minus[dim_i] = max(-5, x[dim_i] - step)
                        x_plus[dim_i] = min(5, x[dim_i] + step)
                        f_minus = counted_func(x_minus)
                        f_plus = counted_func(x_plus)
                
                # Now do golden section search if we have a bracket
                if f_minus < f_curr or f_plus < f_curr:
                    # Determine bracket direction
                    if f_minus < f_plus:
                        a, b = x_minus[dim_i], x[dim_i]
                        fa, fb = f_minus, f_curr
                    else:
                        a, b = x[dim_i], x_plus[dim_i]
                        fa, fb = f_curr, f_plus
                    
                    # Golden section search
                    for _ in range(20):
                        if call_count >= budget or time.time() > deadline:
                            break
                        
                        c = b - inv_phi * (b - a)
                        d = a + inv_phi * (b - a)
                        
                        x_c = x.copy()
                        x_c[dim_i] = c
                        x_d = x.copy()
                        x_d[dim_i] = d
                        
                        fc = counted_func(x_c)
                        fd = counted_func(x_d)
                        
                        if fc < fd:
                            b = d
                            fb = fd
                        else:
                            a = c
                            fa = fc
                    
                    # Update x with the best point found
                    x_best_coord = (a + b) / 2
                    x_best_coord = np.clip(x_best_coord, -5, 5)
                    x_new = x.copy()
                    x_new[dim_i] = x_best_coord
                    f_new = counted_func(x_new)
                    
                    if f_new < f_curr:
                        x = x_new
                        f_curr = f_new
                        best_x = x.copy()
                        best_f = f_curr

        return best_x, best_f
# EVOLVE-BLOCK-END
