import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        call_count = 0
        best_x = None
        best_f = np.inf
        self._best_x = None
        self._best_f = np.inf

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget or time.time() > deadline - 2:
                raise ValueError("Budget or time exceeded")
            call_count += 1
            x_arr = np.asarray(x, dtype=float)
            val = float(func(x_arr))
            if np.isfinite(val) and val < best_f:
                best_f = val
                best_x = np.array(x_arr, dtype=float)
                self._best_x = best_x.copy()
                self._best_f = best_f
            return val

        # Phase 1: Quick coarse sampling to find a good starting region
        # Rosenbrock in 5D has the minimum at (1,1,1,1,1) in standard form,
        # but with hidden shifts/rotations we need to search broadly.
        # Use a mix of structured and random points.
        
        # Try points near the standard Rosenbrock minimum and its neighbors
        standard_min = np.ones(self.dim)
        candidates = [standard_min.copy()]
        
        # Add points around the standard minimum
        for scale in [0.5, 1.0, 2.0, 3.0]:
            for _ in range(5):
                candidates.append(standard_min + rng.normal(0, scale, self.dim))
        
        # Add random points in the box
        for _ in range(30):
            candidates.append(rng.uniform(-4, 4, self.dim))
        
        # Add some axis-aligned points
        for i in range(self.dim):
            for v in [-3, -2, -1, 0, 1, 2, 3]:
                p = np.zeros(self.dim)
                p[i] = v
                candidates.append(p)
        
        # Evaluate candidates
        for p in candidates:
            if call_count >= self.budget or time.time() > deadline - 15:
                break
            try:
                safe_func(p)
            except ValueError:
                break
            except Exception:
                continue

        if best_x is None:
            return np.zeros(self.dim), 0.0

        # Phase 2: Differential Evolution for global search
        # DE is robust for multimodal problems and can handle the narrow valley
        remaining = self.budget - call_count
        if remaining > 50 and time.time() < deadline - 30:
            try:
                def de_callback(xk, convergence):
                    nonlocal call_count, best_x, best_f
                    # Update best from DE's internal tracking
                    if hasattr(xk, '__len__'):
                        pass  # xk is the current population, not a single point
                    return False  # Don't stop early
                
                res_de = differential_evolution(
                    safe_func,
                    bounds=[(-5, 5)] * self.dim,
                    maxiter=50,
                    popsize=15,
                    tol=1e-12,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    seed=42,
                    polish=True,
                    init='sohs',
                    update_interval=1,
                    callback=None,
                    x0=None,
                    constraints=(),
                    integrality=None,
                    vectorized=False
                )
                if np.isfinite(res_de.fun) and res_de.fun < best_f:
                    best_f = float(res_de.fun)
                    best_x = np.array(res_de.x, dtype=float)
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        # Phase 3: Powell's method from the best point found
        remaining = self.budget - call_count
        if remaining > 20 and time.time() < deadline - 15:
            try:
                res_powell = minimize(
                    safe_func,
                    best_x,
                    method='Powell',
                    options={'maxiter': 200, 'maxfev': min(remaining, 100), 
                             'ftol': 1e-14, 'xtol': 1e-12}
                )
                if np.isfinite(res_powell.fun) and res_powell.fun < best_f:
                    best_f = float(res_powell.fun)
                    best_x = np.array(res_powell.x, dtype=float)
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        # Phase 4: L-BFGS-B polish for high precision
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 10:
            try:
                res_lbfgs = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 500, 'maxfun': min(remaining, 50), 
                             'maxls': 50, 'gtol': 1e-14, 'ftol': 1e-16}
                )
                if np.isfinite(res_lbfgs.fun) and res_lbfgs.fun < best_f:
                    best_f = float(res_lbfgs.fun)
                    best_x = np.array(res_lbfgs.x, dtype=float)
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        # Phase 5: Final Nelder-Mead polish with tight tolerances
        remaining = self.budget - call_count
        if remaining > 5 and time.time() < deadline - 5:
            try:
                res_nm = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'maxfev': min(remaining, 20),
                             'fatol': 1e-16, 'xatol': 1e-14}
                )
                if np.isfinite(res_nm.fun) and res_nm.fun < best_f:
                    best_f = float(res_nm.fun)
                    best_x = np.array(res_nm.x, dtype=float)
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        # Phase 6: If we still have budget, try a few more local refinements
        remaining = self.budget - call_count
        if remaining > 5 and time.time() < deadline - 3:
            for _ in range(3):
                if remaining < 5 or time.time() > deadline - 2:
                    break
                try:
                    perturb = rng.normal(0, 0.1, self.dim)
                    start = np.clip(best_x + perturb, -5, 5)
                    res_local = minimize(
                        safe_func,
                        start,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={'maxiter': 100, 'maxfun': min(remaining, 10), 
                                 'gtol': 1e-15, 'ftol': 1e-16}
                    )
                    if np.isfinite(res_local.fun) and res_local.fun < best_f:
                        best_f = float(res_local.fun)
                        best_x = np.array(res_local.x, dtype=float)
                        self._best_x = best_x.copy()
                        self._best_f = best_f
                    remaining = self.budget - call_count
                except Exception:
                    break

        return best_x, best_f
