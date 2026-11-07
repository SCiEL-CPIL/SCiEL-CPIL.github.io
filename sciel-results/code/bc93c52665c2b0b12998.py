import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        n = self.dim
        budget = self.budget
        total_calls = 0
        best_x = np.zeros(n)
        best_f = np.inf

        def try_eval(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return None
            if time.time() > deadline - 5:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_vector(index, dims):
            point = []
            bases = [2, 3, 5, 7, 11]
            for d in range(dims):
                point.append(halton(index, bases[d]))
            return np.array(point)

        # Phase 1: Global Search - sample more broadly
        n_phase1 = 40
        phase1_results = []
        for i in range(1, n_phase1 + 1):
            if total_calls >= budget or time.time() > deadline - 10:
                break
            h_point = halton_vector(i, n)
            x_try = -4.5 + 9.0 * h_point
            f_try = try_eval(x_try)
            if f_try is None:
                break
            phase1_results.append((f_try, x_try.copy()))
        
        # Add structured points
        structured = [np.zeros(n), np.ones(n)*2.0, -np.ones(n)*2.0, 
                      np.ones(n)*-2.0, np.ones(n)*3.0, -np.ones(n)*3.0,
                      np.ones(n)*1.0, -np.ones(n)*1.0]
        for p in structured:
            if total_calls >= budget or time.time() > deadline - 10:
                break
            f_s = try_eval(p)
            if f_s is not None:
                phase1_results.append((f_s, p.copy()))

        if not phase1_results:
            return best_x, best_f

        phase1_results.sort(key=lambda x: x[0])
        
        # Select top candidates for local refinement
        num_candidates = min(8, len(phase1_results))
        candidates = [phase1_results[i][1] for i in range(num_candidates)]
        
        # Phase 2: Local Refinement using L-BFGS-B (gradient-based, more efficient)
        remaining_budget = budget - total_calls
        if remaining_budget <= 0:
            return best_x, best_f
            
        budget_per_candidate = max(20, remaining_budget // num_candidates)
        
        for i, start_x in enumerate(candidates):
            if total_calls >= budget or time.time() > deadline - 10:
                break
                
            def wrapped_func(x):
                f = try_eval(x)
                if f is None:
                    return 1e10
                return f

            try:
                res = minimize(wrapped_func, start_x, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0)] * n,
                               options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-8})
                if hasattr(res, 'fun') and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Nelder-Mead polish on the best found
        if total_calls < budget - 10 and time.time() < deadline - 5:
            def wrapped_func_polish(x):
                f = try_eval(x)
                if f is None:
                    return 1e10
                return f

            try:
                res = minimize(wrapped_func_polish, best_x, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
                if hasattr(res, 'fun') and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 4: Final coordinate-wise refinement
        if total_calls < budget - 5 and time.time() < deadline - 3:
            for _ in range(3):
                if total_calls >= budget or time.time() > deadline - 3:
                    break
                improved = True
                while improved:
                    improved = False
                    for d in range(n):
                        if total_calls >= budget or time.time() > deadline - 3:
                            break
                        # Try small perturbations in each direction
                        for delta in [0.1, -0.1, 0.01, -0.01, 0.001, -0.001]:
                            if total_calls >= budget or time.time() > deadline - 3:
                                break
                            x_try = best_x.copy()
                            x_try[d] += delta
                            x_try = np.clip(x_try, -5.0, 5.0)
                            f_try = try_eval(x_try)
                            if f_try is not None and f_try < best_f:
                                improved = True
                                break

        return best_x, best_f
