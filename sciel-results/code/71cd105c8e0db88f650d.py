import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 310

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        eval_count = 0
        best_x = None
        best_f = np.inf
        candidates = []

        def counted_func(x):
            nonlocal eval_count, best_f, best_x
            if time.time() > self.deadline:
                raise TimeoutError
            if eval_count >= budget:
                raise RuntimeError("Budget exceeded")
            
            x = np.asarray(x, dtype=float)
            x_clipped = np.clip(x, -5.0, 5.0)
            
            eval_count += 1
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            
            candidates.append((fx, x_clipped.copy()))
            
            return fx

        # Phase 1: Global Search
        # Use a smaller number of global samples to start local optimization sooner.
        # 10 samples is often enough to find the basin of attraction for Rosenbrock in 5D.
        n_global = 10
        for i in range(n_global):
            if time.time() > self.deadline or eval_count >= budget:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            try:
                counted_func(x)
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                continue

        if best_x is None:
            return np.zeros(dim), np.inf

        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Local Refinement with L-BFGS-B
        bounds = [(-5.0, 5.0)] * dim
        
        # Try optimizing from the top candidates.
        # We prioritize the best candidate first to ensure early improvement in AOCC.
        # If it fails or doesn't improve much, we try the next best.
        top_candidates = candidates[:5]
        
        for idx, (f_val, x_start) in enumerate(top_candidates):
            if time.time() > self.deadline or eval_count >= budget - 5:
                break
            
            remaining_budget = budget - eval_count
            if remaining_budget <= 10:
                break
                
            try:
                # L-BFGS-B is efficient. We give it a generous maxfun but it usually converges quickly.
                res = scipy.optimize.minimize(
                    counted_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxfun': remaining_budget, 'maxiter': 1000, 'gtol': 1e-8}
                )
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                continue
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x

            # If we've already used a significant portion of the budget, we might stop
            # to allow for a final polish or just to save time.
            # However, for Rosenbrock, one good L-BFGS-B run is usually sufficient.
            # We break if we've made significant progress or if we're running low on budget.
            if eval_count > budget * 0.5:
                break

        return best_x, best_f
