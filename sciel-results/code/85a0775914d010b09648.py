"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start (1+1)-ES with 1/5 success rule for ill-conditioned problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _es_1plus1(self, func, rng, x0, sigma0, max_calls, deadline):
        """
        (1+1)-ES with 1/5 success rule.
        Returns best_x, best_f, calls_used.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        sigma = sigma0
        calls = 1
        
        # 1/5 success rule parameters
        # Increase sigma by factor c_inc if success, decrease by c_dec if failure
        # over a window of n_window steps
        c_inc = 1.05
        c_dec = 0.95
        n_window = 10
        success_count = 0
        
        best_x = x.copy()
        best_f = fx
        
        while calls < max_calls and time.time() < deadline:
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x = y
                fx = fy
                success_count += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                success_count -= 1 if success_count > 0 else 0
            
            # Adjust sigma based on success rate over window
            # Simple approach: adjust every n_window steps
            if calls % n_window == 0:
                success_rate = success_count / n_window
                if success_rate > 0.5:
                    sigma *= c_inc
                elif success_rate < 0.5:
                    sigma *= c_dec
                success_count = 0
            
            sigma = np.clip(sigma, 1e-10, 10.0)
        
        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(123)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0
        
        # Phase 1: Initial random sampling to find good starting points
        n_samples = min(20, budget // 10)
        samples = rng.uniform(-4, 4, size=(n_samples, dim))
        sample_values = []
        for i in range(n_samples):
            if total_calls >= budget or time.time() > deadline - 5.0:
                break
            x = samples[i]
            fx = func(x)
            total_calls += 1
            sample_values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Multi-start (1+1)-ES
        # Use top-k samples as starting points
        if len(sample_values) > 0:
            sorted_idx = np.argsort(sample_values)
            n_starts = min(5, len(sorted_idx))
            
            # Allocate remaining budget among starts
            remaining_budget = budget - total_calls
            budget_per_start = max(10, remaining_budget // (n_starts + 1))
            
            for start_i in range(n_starts):
                if total_calls >= budget or time.time() > deadline - 5.0:
                    break
                
                x0 = samples[sorted_idx[start_i]].copy()
                # Estimate initial sigma from sample spread
                sigma0 = max(0.5, np.std(samples, axis=0).mean())
                
                x_best, f_best, calls_used = self._es_1plus1(
                    func, rng, x0, sigma0, budget_per_start, deadline
                )
                total_calls += calls_used
                
                if f_best < best_f:
                    best_f = f_best
                    best_x = x_best.copy()
        
        # Phase 3: Final local polish from the best point
        remaining = budget - total_calls
        if remaining > 5 and time.time() < deadline - 5.0:
            x_local = best_x.copy()
            fx_local = best_f
            sigma_local = 0.1
            
            local_calls = 0
            while local_calls < remaining and time.time() < deadline - 5.0:
                z = rng.standard_normal(dim)
                y = x_local + sigma_local * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                local_calls += 1
                total_calls += 1
                
                if fy < fx_local:
                    x_local = y
                    fx_local = fy
                    sigma_local *= 1.1
                    if fx_local < best_f:
                        best_f = fx_local
                        best_x = x_local.copy()
                else:
                    sigma_local *= 0.9
                
                sigma_local = np.clip(sigma_local, 1e-8, 10.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
