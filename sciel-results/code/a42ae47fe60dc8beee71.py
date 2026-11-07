import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.start_time = time.time()
        self.time_limit = 290.0  # Safety margin before 330s kill

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        self.best_x = None
        self.best_f = float('inf')
        self._last_update_time = self.start_time

        def wrapped_func(x):
            self.call_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > self.time_limit:
                raise TimeoutError("Time limit exceeded")
            x = np.asarray(x, dtype=np.float64).copy()
            x = np.clip(x, -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                f = float('inf')
            if np.isfinite(f) and f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
                self._last_update_time = time.time()
            return f

        def time_remaining():
            return self.time_limit - (time.time() - self.start_time)

        def calls_remaining():
            return self.budget - self.call_count

        n = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0) for _ in range(n)]

        # Halton sequence for deterministic low-discrepancy initialization
        def halton(dim, n_points, skip=1):
            points = np.zeros((n_points, dim))
            for d in range(dim):
                base = d + 2  # primes: 2, 3, 5, 7, 11
                h = np.zeros(n_points)
                i = skip
                for idx in range(n_points):
                    f = 1.0
                    r = 0.0
                    i_val = i + idx
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    h[idx] = r
                points[:, d] = h
            return points

        # Phase 1: Multi-start with Halton sequence for diverse coverage
        n_starts = 25
        X0 = halton(n, n_starts, skip=2)
        X0 = X0 * 10.0 - 5.0  # Scale to [-5, 5]

        # Evaluate initial points
        candidates = []
        for x in X0:
            if self.call_count >= budget or time_remaining() < 5.0:
                break
            try:
                f = wrapped_func(x)
                if np.isfinite(f):
                    candidates.append((f, x.copy()))
            except Exception:
                pass

        if not candidates:
            return self.best_x, self.best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Multi-start L-BFGS-B with Nelder-Mead polish
        # Use top candidates for local refinement
        top_k = min(8, len(candidates))
        
        for i in range(top_k):
            if self.call_count >= budget - 10 or time_remaining() < 30.0:
                break
            
            f0, x0 = candidates[i]
            
            # L-BFGS-B for gradient-based local optimization
            try:
                maxfun = max(10, min(80, (budget - self.call_count - 10) // max(1, top_k - i)))
                res_lbfgs = minimize(wrapped_func, x0, method='L-BFGS-B', bounds=bounds,
                                     options={'maxfun': maxfun, 'maxiter': 200,
                                              'ftol': 1e-12, 'gtol': 1e-8})
                if res_lbfgs.fun < self.best_f:
                    self.best_f = res_lbfgs.fun
                    self.best_x = np.clip(res_lbfgs.x, -5.0, 5.0).copy()
            except Exception:
                pass

            # Nelder-Mead polish for derivative-free refinement
            if self.call_count < budget - 5 and time_remaining() > 10.0:
                try:
                    start_nelder = np.clip(self.best_x, -5.0, 5.0) if self.best_x is not None else x0
                    maxiter_nelder = max(10, min(50, (budget - self.call_count - 5) // max(1, top_k - i)))
                    res_nm = minimize(wrapped_func, start_nelder, method='Nelder-Mead',
                                      options={'maxiter': maxiter_nelder, 'maxfev': maxiter_nelder * 2,
                                               'xatol': 1e-10, 'fatol': 1e-12})
                    if res_nm.fun < self.best_f:
                        self.best_f = res_nm.fun
                        self.best_x = np.clip(res_nm.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        # Phase 3: Final intensive polish on the absolute best
        if self.call_count < budget - 3 and self.best_x is not None and time_remaining() > 15.0:
            try:
                # One more L-BFGS-B pass
                remaining_calls = budget - self.call_count - 3
                res_final = minimize(wrapped_func, self.best_x, method='L-BFGS-B', bounds=bounds,
                                     options={'maxfun': remaining_calls // 2, 'maxiter': 100,
                                              'ftol': 1e-14, 'gtol': 1e-10})
                if res_final.fun < self.best_f:
                    self.best_f = res_final.fun
                    self.best_x = np.clip(res_final.x, -5.0, 5.0).copy()
            except Exception:
                pass

            # Final Nelder-Mead polish
            if self.call_count < budget - 2 and time_remaining() > 5.0 and self.best_x is not None:
                try:
                    remaining_calls = budget - self.call_count - 2
                    res_final_nm = minimize(wrapped_func, self.best_x, method='Nelder-Mead',
                                            options={'maxiter': remaining_calls, 'maxfev': remaining_calls * 2,
                                                     'xatol': 1e-12, 'fatol': 1e-14})
                    if res_final_nm.fun < self.best_f:
                        self.best_f = res_final_nm.fun
                        self.best_x = np.clip(res_final_nm.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        return self.best_x, self.best_f
