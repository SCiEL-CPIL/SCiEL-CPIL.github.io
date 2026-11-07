import numpy as np
import scipy.optimize as opt
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        used_calls = 0
        max_calls = self.budget
        
        def safe_func(x):
            nonlocal used_calls
            if used_calls >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            used_calls += 1
            return func(np.clip(x, -5, 5))
        
        # Phase 1: Structured global exploration
        # Use Latin Hypercube for uniform coverage + some Sobol points
        n_lhc = 20
        n_sobol = 15
        n_total_initial = n_lhc + n_sobol
        n_total_initial = min(n_total_initial, max_calls // 4)
        
        # Latin Hypercube sampling for uniform coverage
        rng_lhc = np.random.default_rng(123)
        lhc_points = np.zeros((n_lhc, self.dim))
        for d in range(self.dim):
            perm = rng_lhc.permutation(n_lhc)
            lhc_points[:, d] = (perm + 0.5) / n_lhc
        
        # Sobol sequence for low-discrepancy coverage
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_sobol)
        
        # Combine and scale to [-5, 5]^d
        initial_points = np.vstack([lhc_points, sobol_points])
        initial_points = initial_points * 10 - 5
        
        # Evaluate initial points
        initial_values = []
        for p in initial_points:
            if used_calls >= max_calls or time.time() > deadline:
                break
            try:
                fp = safe_func(p)
                initial_values.append((fp, p))
                if fp < best_f:
                    best_f = fp
                    best_x = p.copy()
            except RuntimeError:
                break
        
        if best_x is None:
            return None, np.inf
        
        # Phase 2: CMA-ES with adaptive restarts
        initial_values.sort(key=lambda x: x[0])
        
        remaining_budget = max_calls - used_calls
        
        # Use top candidates for restarts
        n_candidates = min(5, len(initial_values))
        candidates = initial_values[:n_candidates]
        
        # CMA-ES parameters
        sigma0 = 1.5  # Initial step size
        
        for fp, x_start in candidates:
            if used_calls >= max_calls or time.time() > deadline:
                break
            
            local_budget = max_calls - used_calls
            if local_budget < 30:
                break
            
            try:
                # CMA-ES is more robust for multimodal functions
                result = opt.minimize(
                    safe_func, 
                    x_start, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': local_budget,
                        'maxfev': local_budget,
                        'xatol': 1e-7,
                        'fatol': 1e-7,
                        'adaptive': True
                    }
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
            except (RuntimeError, Exception):
                if time.time() > deadline or used_calls >= max_calls:
                    break
                continue
        
        # Phase 3: Fine-tuning around the best solution
        if used_calls < max_calls and time.time() < deadline:
            final_budget = max_calls - used_calls
            if final_budget > 15:
                try:
                    result = opt.minimize(
                        safe_func, 
                        best_x, 
                        method='Nelder-Mead',
                        options={
                            'maxiter': final_budget,
                            'maxfev': final_budget,
                            'xatol': 1e-9,
                            'fatol': 1e-9,
                            'adaptive': True
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except (RuntimeError, Exception):
                    pass
        
        return best_x, best_f
