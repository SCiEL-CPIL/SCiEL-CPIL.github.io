import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Optimizer using random initialization, coordinate descent with golden section, and Powell polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Random sampling
        rng = np.random.default_rng()
        n_samples = 30
        for i in range(n_samples):
            if time.time() > deadline or call_count >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            wrapped_func(x)

        if best_x is None:
            best_x = np.zeros(self.dim)
            wrapped_func(best_x)

        # Phase 2: Coordinate Descent with Golden Section
        # This exploits the anisotropy of the Bent Cigar function.
        # We iterate over dimensions, performing 1D minimization on each.
        
        def golden_section_1d(x, dim):
            nonlocal call_count, best_x, best_f
            if time.time() > deadline or call_count >= self.budget:
                return x
            
            a = -5.0
            b = 5.0
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            x_c = x.copy()
            x_c[dim] = c
            f_c = wrapped_func(x_c)
            
            x_d = x.copy()
            x_d[dim] = d
            f_d = wrapped_func(x_d)
            
            # Iterate until interval is small or budget/time runs out
            while (b - a) > 1e-8 and time.time() < deadline and call_count < self.budget:
                if f_c < f_d:
                    b = d
                    d = c
                    f_d = f_c
                    c = b - gr * (b - a)
                    x_c = x.copy()
                    x_c[dim] = c
                    f_c = wrapped_func(x_c)
                else:
                    a = c
                    c = d
                    f_c = f_d
                    d = a + gr * (b - a)
                    x_d = x.copy()
                    x_d[dim] = d
                    f_d = wrapped_func(x_d)
            
            # Return the midpoint of the final interval
            x_opt = x.copy()
            x_opt[dim] = (a + b) / 2.0
            return x_opt

        # Run multiple passes of coordinate descent
        # Each pass iterates through all dimensions
        # We stop if no significant improvement is made or budget runs out
        max_cd_passes = 10
        for _ in range(max_cd_passes):
            if time.time() > deadline or call_count >= self.budget:
                break
            
            x_current = best_x.copy()
            improved = False
            for dim in range(self.dim):
                if time.time() > deadline or call_count >= self.budget:
                    break
                x_new = golden_section_1d(x_current, dim)
                # Check if we found a better point
                f_new = wrapped_func(x_new)
                if f_new < best_f:
                    best_x = x_new.copy()
                    best_f = f_new
                    improved = True
            
            if not improved:
                break

        # Phase 3: Final polish with Powell if budget remains
        remaining_budget = self.budget - call_count
        if remaining_budget > 20 and time.time() < deadline:
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='Powell',
                    options={'maxiter': 500, 'maxfev': remaining_budget, 'ftol': 1e-15, 'xtol': 1e-15, 'disp': False}
                )
            except Exception:
                pass

        return best_x, best_f
