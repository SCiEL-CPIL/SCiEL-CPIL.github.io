"""BFGS + Powell optimizer exploiting quadratic structure of BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import fmin_bfgs, fmin_powell


class Optimizer:
    """BFGS and Powell optimizer for the BBOB-style Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_remaining():
            return deadline - time.time()

        def make_scipy_func(x0):
            """Create a function wrapper that tracks budget and clips."""
            def scipy_func(x):
                if call_count >= budget or time.time() > deadline - 5:
                    raise RuntimeError("Budget or time exhausted")
                x_clipped = np.clip(x, -5.0, 5.0)
                fx = func(x_clipped)
                call_count += 1
                return fx
            return scipy_func

        # Phase 1: Random sampling to find promising regions
        # Sample more broadly to find the basin of attraction
        n_samples = min(25, budget // 10)
        sample_results = []
        for _ in range(n_samples):
            if call_count >= budget or time.time() > deadline - 100:
                break
            x_sample = rng.uniform(-4.5, 4.5, dim)
            fx = try_eval(x_sample)
            if fx is None:
                break
            sample_results.append((x_sample.copy(), fx))
        
        # Also evaluate origin
        if call_count < budget and time.time() <= deadline - 100:
            fx_origin = try_eval(np.zeros(dim))
            if fx_origin is not None:
                sample_results.append((np.zeros(dim), fx_origin))
        
        # Sort samples by function value
        sample_results.sort(key=lambda x: x[1])
        
        # Phase 2: Apply BFGS to top starting points
        # BFGS is efficient for smooth convex functions like Discus
        n_restarts = min(3, len(sample_results))
        
        for idx in range(n_restarts):
            if call_count >= budget or time.time() > deadline - 30:
                break
            
            x_start = sample_results[idx][0].copy()
            
            # Skip if too close to current best
            if best_x is not None and np.linalg.norm(x_start - best_x) < 0.3:
                continue
            
            # Use BFGS with maxiter based on remaining budget
            remaining = budget - call_count
            maxiter = min(remaining // 2, 100)
            
            if maxiter < 5:
                continue
            
            try:
                scipy_func = make_scipy_func(x_start)
                # BFGS for smooth optimization
                x_opt, f_opt, info = fmin_bfgs(scipy_func, x_start, maxiter=maxiter, full_output=True, disp=False)
                # Update best
                x_opt = np.clip(x_opt, -5.0, 5.0)
                f_final = func(x_opt)
                call_count += 1
                if f_final < best_f:
                    best_f = f_final
                    best_x = x_opt.copy()
            except (RuntimeError, Exception):
                # Budget or time exhausted, continue with what we have
                pass
        
        # Phase 3: Fine-tune with Powell method (derivative-free, good for final polish)
        if best_x is not None and call_count < budget and time.time() < deadline - 20:
            remaining = budget - call_count
            if remaining >= 10:
                try:
                    scipy_func = make_scipy_func(best_x)
                    x_powell, f_powell = fmin_powell(scipy_func, best_x, maxiter=min(remaining // 2, 50), disp=False)
                    x_powell = np.clip(x_powell, -5.0, 5.0)
                    f_final = func(x_powell)
                    call_count += 1
                    if f_final < best_f:
                        best_f = f_final
                        best_x = x_powell.copy()
                except (RuntimeError, Exception):
                    pass
        
        # Phase 4: Final local refinement with small random steps
        if best_x is not None and call_count < budget and time.time() < deadline - 10:
            sigma = 0.1
            for _ in range(20):
                if call_count >= budget or time.time() > deadline - 5:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < best_f:
                    best_x = y.copy()
                    best_f = fy
                    sigma *= 1.1
                else:
                    sigma *= 0.7

        # Ensure we return something valid
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
