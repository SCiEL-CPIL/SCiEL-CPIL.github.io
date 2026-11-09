import numpy as np
import time
from scipy.optimize import minimize, dual_annealing

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                raise TimeoutError()
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if not np.isfinite(fx):
                fx = float('inf')
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global exploration with Dual Annealing
        # Dual Annealing is robust for Rosenbrock's multi-modal landscape
        da_budget = int(budget * 0.4)  # 40% of budget for global search
        if da_budget > 20:
            try:
                bounds = [(-4.0, 4.0)] * dim
                da_result = dual_annealing(
                    safe_func,
                    bounds,
                    maxiter=50,
                    maxfev=da_budget,
                    seed=42,
                    restarts=3
                )
            except Exception:
                pass

        if not np.isfinite(best_f):
            rng = np.random.default_rng(123)
            for _ in range(min(50, budget - 10)):
                if time.time() > deadline:
                    break
                x = rng.uniform(-4, 4, dim)
                safe_func(x)

        if not np.isfinite(best_f):
            return best_x, best_f

        # Phase 2: Adaptive line search along Halton directions
        # This exploits the anisotropic valley structure of Rosenbrock
        remaining = budget - calls
        if remaining > 20 and time.time() < deadline:
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r

            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            n_halton = min(remaining // 15, 20)  # Number of directions to try
            
            for i in range(1, n_halton + 1):
                if remaining < 10 or time.time() > deadline:
                    break
                
                # Generate Halton direction
                d = np.array([halton(i, p) for p in primes[:dim]])
                d = 2.0 * d - 1.0  # Map to [-1, 1]
                norm = np.linalg.norm(d)
                if norm < 1e-10:
                    continue
                d /= norm
                
                # Golden-section search along this direction
                phi = (1 + np.sqrt(5)) / 2
                inv_phi = 1.0 / phi
                inv_phi2 = 1.0 / (phi * phi)
                
                # Search range: [-3, 3] along the direction (centered at best_x)
                a = -3.0
                b = 3.0
                c = b - inv_phi * (b - a)
                d_pt = a + inv_phi * (b - a)
                
                xc = best_x + c * d
                xc = np.clip(xc, -5.0, 5.0)
                fc = safe_func(xc)
                
                xd = best_x + d_pt * d
                xd = np.clip(xd, -5.0, 5.0)
                fd = safe_func(xd)
                
                # Iterate golden section search
                max_iter = min(remaining - 4, 12)
                for _ in range(max_iter):
                    if remaining < 4 or time.time() > deadline:
                        break
                    
                    if fc < fd:
                        b = d_pt
                        d_pt = c
                        fd = fc
                        c = b - inv_phi * (b - a)
                        xc = best_x + c * d
                        xc = np.clip(xc, -5.0, 5.0)
                        fc = safe_func(xc)
                    else:
                        a = c
                        c = d_pt
                        fc = fd
                        d_pt = a + inv_phi * (b - a)
                        xd = best_x + d_pt * d
                        xd = np.clip(xd, -5.0, 5.0)
                        fd = safe_func(xd)

        # Phase 3: Local refinement with L-BFGS-B
        local_budget = budget - calls
        if local_budget > 10:
            try:
                result = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 200,
                        'maxfun': local_budget,
                        'gtol': 1e-12,
                        'ftol': 1e-12
                    }
                )
            except Exception:
                pass

        # Phase 4: Final high-precision polish with Powell
        remaining = budget - calls
        if remaining > 5 and time.time() < deadline:
            try:
                result = minimize(
                    safe_func,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 50,
                        'maxfev': remaining,
                        'xtol': 1e-12,
                        'ftol': 1e-12
                    }
                )
            except Exception:
                pass

        return best_x, best_f
