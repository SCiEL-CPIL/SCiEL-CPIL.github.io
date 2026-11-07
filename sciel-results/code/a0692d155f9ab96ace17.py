import numpy as np
import time
import scipy.optimize
import warnings

warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        
        call_count = 0
        best_x = None
        best_fx = np.inf

        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            if time.time() > deadline:
                raise TimeoutError
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Exploration
        # 50 random points in [-4, 4]^5
        n_global = 50
        candidates = []
        for _ in range(n_global):
            candidates.append(rng.uniform(-4, 4, self.dim))
        
        # Also include origin and ones just in case
        candidates.append(np.zeros(self.dim))
        candidates.append(np.ones(self.dim))
        
        # Evaluate
        for x0 in candidates:
            if call_count >= self.budget - 50:
                break
            try:
                tracked_func(x0)
            except TimeoutError:
                break
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            tracked_func(best_x)

        # Phase 2: Local Optimization with L-BFGS-B
        # L-BFGS-B is robust and efficient for smooth functions.
        bounds = [(-5, 5)] * self.dim
        remaining_budget = self.budget - call_count - 10
        
        if remaining_budget > 10:
            try:
                res = scipy.optimize.minimize(
                    tracked_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 1000,
                        'maxfun': remaining_budget,
                        'ftol': 1e-15,
                        'gtol': 1e-15,
                        'eps': 1e-6 # Slightly larger eps for stability?
                    }
                )
                # tracked_func updates best_x/best_fx automatically
            except TimeoutError:
                pass
            except Exception:
                pass

        # Phase 3: Polish with Powell if budget remains?
        # L-BFGS-B might stop early due to gradient tolerance.
        # Powell can squeeze out more.
        if time.time() < deadline and call_count < self.budget - 5:
            remaining = self.budget - call_count - 5
            if remaining > 5:
                try:
                    res = scipy.optimize.minimize(
                        tracked_func,
                        best_x,
                        method='Powell',
                        bounds=bounds,
                        options={
                            'maxfev': remaining,
                            'xtol': 1e-12,
                            'ftol': 1e-12
                        }
                    )
                except TimeoutError:
                    pass
                except Exception:
                    pass

        return best_x, best_fx
