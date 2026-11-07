import numpy as np
import time
import warnings
from scipy.optimize import minimize, differential_evolution

warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)
        
        d = self.dim
        lb = np.full(d, -5.0)
        ub = np.full(d, 5.0)
        bounds = [(-5.0, 5.0)] * d

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
                
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, lb, ub)
            
            try:
                f = float(func(x))
                if not np.isfinite(f):
                    f = np.inf
            except Exception:
                f = np.inf
                
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Strategy:
        # 1. Short global sampling to establish diverse baselines.
        # 2. Aggressive local refinement using L-BFGS-B and Nelder-Mead.
        # 3. If budget remains, use Differential Evolution for robust global search.
        # 4. Final polish on the global best.

        # Phase 1: Global Sampling
        # 30 samples is sufficient for 5D to hit multiple basins, saving budget for local.
        num_global = 30
        candidates = []
        for i in range(num_global):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, d)
            f = call_func(x)
            candidates.append((f, x))
        
        # Sort candidates by fitness
        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Local Refinement
        # Run local optimization from the top candidates.
        max_local_runs = 10
        
        for i in range(max_local_runs):
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            # Select start point
            if i < len(candidates):
                _, x_start = candidates[i]
            else:
                x_start = rng.uniform(-5.0, 5.0, d)
                call_func(x_start)
            
            # Dynamic budget allocation
            remaining = self.budget - calls_made
            runs_left = max_local_runs - i
            if runs_left <= 0:
                runs_left = 1
            
            # Target ~20 calls per L-BFGS-B run, but scale with remaining budget
            target_per_run = 20
            maxfev_lbfgs = int(min(remaining * 0.7, max(15, target_per_run * (remaining // (runs_left * 1.5)))))
            maxfev_lbfgs = max(15, min(60, maxfev_lbfgs))
            
            # Run L-BFGS-B
            res_x = x_start
            try:
                res = minimize(
                    call_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxfev': maxfev_lbfgs,
                        'ftol': 1e-14,
                        'gtol': 1e-10
                    }
                )
                res_x = res.x
            except Exception:
                pass
            
            # Polish with Nelder-Mead
            if calls_made < self.budget and time.time() < deadline:
                remaining_iter = self.budget - calls_made
                nm_maxfev = max(10, min(20, remaining_iter // max(1, runs_left)))
                
                try:
                    res_nm = minimize(
                        call_func,
                        res_x,
                        method='Nelder-Mead',
                        options={
                            'maxfev': nm_maxfev,
                            'xatol': 1e-10,
                            'fatol': 1e-10
                        }
                    )
                except Exception:
                    pass
        
        # Phase 3: Differential Evolution if significant budget remains
        # DE is robust and can explore the space more thoroughly than random sampling.
        if calls_made < self.budget - 50 and time.time() < deadline and best_x is not None:
            remaining = self.budget - calls_made
            # Allocate up to 100 calls to DE, but not more than half the remaining budget
            de_maxiter = min(100, remaining // 2)
            if de_maxiter > 10:
                try:
                    def de_wrapper(x):
                        return call_func(x)
                    
                    res_de = differential_evolution(
                        de_wrapper,
                        bounds,
                        maxiter=de_maxiter,
                        popsize=15,
                        tol=1e-10,
                        seed=42,
                        polish=False,  # We do our own polish
                        init='latinhypercube'
                    )
                except Exception:
                    pass
        
        # Phase 4: Final Polish
        # If we have budget left, do one more intensive run from the global best.
        if calls_made < self.budget and time.time() < deadline and best_x is not None:
            remaining = self.budget - calls_made
            if remaining > 10:
                try:
                    minimize(
                        call_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxfev': min(30, remaining),
                            'ftol': 1e-15,
                            'gtol': 1e-12
                        }
                    )
                except Exception:
                    pass
                
                if calls_made < self.budget and time.time() < deadline:
                    remaining = self.budget - calls_made
                    if remaining > 5:
                        try:
                            minimize(
                                call_func,
                                best_x,
                                method='Nelder-Mead',
                                options={
                                    'maxfev': min(15, remaining),
                                    'xatol': 1e-12,
                                    'fatol': 1e-12
                                }
                            )
                        except Exception:
                            pass

        if best_x is None:
            best_x = np.zeros(d)
            call_func(best_x)

        return best_x, best_f
