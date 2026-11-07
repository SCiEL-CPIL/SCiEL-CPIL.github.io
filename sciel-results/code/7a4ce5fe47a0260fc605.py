import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _katsuura_structure_starts(self, dim, n_starts):
        """Generate diverse starting points based on Katsuura structure.
        
        The Katsuura function has cosine product terms cos(pi * f_d * x_d) 
        and a polynomial term minimized near x_d=1. The global minimum 
        involves finding x where the cosine product is near -1 and the 
        polynomial is small.
        
        We generate candidates by:
        1. Points near (1,1,1,1,1) - polynomial minimum region
        2. Points where individual cosines are near -1: x_d = (2k+1)/f_d
        3. Random diverse points
        """
        starts = []
        
        # Base point: polynomial minimum
        starts.append(np.ones(dim))
        
        # Points with one dimension shifted to cosine minimum
        for d in range(dim):
            for k in [-1, 0, 1, 2]:
                x = np.ones(dim)
                # Try x_d = (2k+1) which gives cos(pi*(2k+1)) = -1 if f_d=1
                x[d] = (2 * k + 1)
                if -5.0 <= x[d] <= 5.0:
                    starts.append(x.copy())
        
        # Points with two dimensions shifted
        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                for k1 in [0, 1]:
                    for k2 in [0, 1]:
                        x = np.ones(dim)
                        x[d1] = 2 * k1 + 1
                        x[d2] = 2 * k2 + 1
                        if -5.0 <= x[d1] <= 5.0 and -5.0 <= x[d2] <= 5.0:
                            starts.append(x.copy())
        
        # Add some random diverse points
        rng = np.random.RandomState(42)
        for _ in range(max(1, n_starts - len(starts))):
            x = rng.uniform(-4, 4, size=dim)
            starts.append(x)
        
        # Deduplicate and clip
        unique_starts = []
        seen = set()
        for x in starts:
            x_c = np.clip(x, -5, 5)
            key = tuple(np.round(x_c, 6))
            if key not in seen:
                seen.add(key)
                unique_starts.append(x_c)
        
        return unique_starts[:n_starts]

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        eval_count = 0

        def counting_func(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x_c = np.clip(np.asarray(x, dtype=float), -5, 5)
            try:
                val = float(func(x_c))
            except Exception:
                return np.inf
            eval_count += 1
            if val < best_f:
                best_f = val
                best_x = x_c.copy()
            return val

        def still_ok():
            return eval_count < budget and time.time() < deadline

        # Phase 1: Evaluate diverse structural starting points
        # Use about 30% of budget for initial exploration
        n_explore = min(15, budget // 3)
        starts = self._katsuura_structure_starts(dim, n_explore)
        
        # Evaluate all starting points
        for x_start in starts:
            if not still_ok():
                break
            counting_func(x_start)
        
        if best_x is None:
            best_x = np.ones(dim)
            counting_func(best_x)
        
        # Phase 2: Multi-start Nelder-Mead refinement
        # Use remaining budget for local optimization from best few starts
        remaining = budget - eval_count
        if remaining > 10 and still_ok():
            # Try Nelder-Mead from the current best
            try:
                res = scipy.optimize.minimize(
                    counting_func,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(10, remaining // 3),
                        'xatol': 1e-8, 
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
            except Exception:
                pass
        
        # Phase 3: If budget remains, try another local method from best
        remaining = budget - eval_count
        if remaining > 8 and still_ok() and best_x is not None:
            try:
                res = scipy.optimize.minimize(
                    counting_func,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': max(5, remaining // 4), 'maxfun': remaining}
                )
            except Exception:
                pass
        
        # Phase 4: Final high-precision polish if budget remains
        remaining = budget - eval_count
        if remaining > 5 and still_ok() and best_x is not None:
            try:
                res = scipy.optimize.minimize(
                    counting_func,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(3, remaining),
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.ones(dim)
            try:
                best_f = float(func(best_x))
            except Exception:
                best_f = 0.0

        return best_x, best_f
