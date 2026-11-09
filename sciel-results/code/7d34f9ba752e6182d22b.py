"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with diverse initialization, 1/5 success rule, and polish phase."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        start_time = time.time()
        time_limit = 325.0

        # Phase 1: Diverse initialization - evaluate several candidates
        n_init = min(10, max(5, budget // 30))
        candidates = []
        
        # Always include center
        x0 = np.zeros(n)
        x0 = np.clip(x0, -5, 5)
        candidates.append(x0)
        
        # Add random points
        rng = np.random.default_rng(42)
        for _ in range(n_init - 1):
            xc = rng.uniform(-4, 4, size=n)
            xc = np.clip(xc, -5, 5)
            candidates.append(xc)
        
        # Evaluate all candidates
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        for cand in candidates:
            if time.time() - start_time > time_limit:
                break
            try:
                fc = func(cand)
                eval_count += 1
                if fc < best_f:
                    best_f = fc
                    best_x = cand.copy()
            except Exception:
                break
        
        if best_x is None:
            # Fallback: single evaluation at origin
            x0 = np.zeros(n)
            x0 = np.clip(x0, -5, 5)
            best_x = x0
            best_f = func(best_x)
            eval_count = 1
        
        x = best_x.copy()
        fx = best_f
        x_best = best_x.copy()
        f_best = best_f
        
        # Phase 2: (1+1)-ES with 1/5 success rule
        sigma = 2.0
        window_size = 15
        successes = []
        
        while eval_count < budget:
            if time.time() - start_time > time_limit:
                break
            
            try:
                z = np.random.standard_normal(n)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
            except Exception:
                break
            
            eval_count += 1
            
            success = fy < fx
            if success:
                x = y
                fx = fy
                x_best = y.copy()
                f_best = fy
            else:
                successes.append(1 if success else 0)
            
            # Adapt sigma using 1/5 success rule on sliding window
            if len(successes) >= window_size:
                successes.pop(0)
            if len(successes) >= 5:
                success_rate = np.mean(successes)
                if success_rate > 0.2:
                    sigma *= 1.25
                elif success_rate < 0.2:
                    sigma *= 0.8
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-12, 100.0)
            
            if sigma < 1e-10:
                break
        
        # Phase 3: Polish with smaller steps if budget remains
        remaining = budget - eval_count
        if remaining > 10 and f_best > -1e6:
            sigma_polish = max(1e-3, sigma * 0.1)
            for _ in range(min(remaining, 50)):
                if time.time() - start_time > time_limit:
                    break
                try:
                    z = np.random.standard_normal(n)
                    y = x + sigma_polish * z
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    eval_count += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        x_best = y.copy()
                        f_best = fy
                        sigma_polish *= 1.1
                    else:
                        sigma_polish *= 0.9
                    sigma_polish = max(sigma_polish, 1e-12)
                except Exception:
                    break
        
        return x_best, f_best
# EVOLVE-BLOCK-END
