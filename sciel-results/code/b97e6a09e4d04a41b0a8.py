"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid Halton + Multi-restart (1+1)-ES optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        # State
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def call_func(x):
            nonlocal calls_made, best_x, best_f
            # Check time limit
            if time.time() > deadline:
                raise TimeoutError()
            
            # Clip to box
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Global Phase: Halton Sequence Sampling
        # Generate low-discrepancy points for good coverage
        n_global = 100
        if self.budget < n_global:
            n_global = self.budget
            
        # Simple Halton generator for 5 dimensions
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        points = []
        for i in range(1, n_global + 1):
            p = []
            for j in range(self.dim):
                # Primes for bases: 2, 3, 5, 7, 11
                base = [2, 3, 5, 7, 11][j]
                p.append(-5 + 10 * halton(i, base))
            points.append(p)
            
        # Evaluate global points
        global_results = []
        for p in points:
            try:
                fx = call_func(p)
                global_results.append((fx, p))
            except TimeoutError:
                break
                
        if calls_made >= self.budget:
            return best_x, best_f

        # Sort by fitness
        global_results.sort(key=lambda x: x[0])
        
        # 2. Local Phase: Multi-restart (1+1)-ES
        # Select top candidates for local refinement
        n_restarts = min(10, len(global_results))
        remaining_budget = self.budget - calls_made
        budget_per_restart = remaining_budget // n_restarts if n_restarts > 0 else 0
        
        for i in range(n_restarts):
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x = np.array(global_results[i][1])
            fx = global_results[i][0]
            
            # Local (1+1)-ES
            sigma = 0.5
            rng = np.random.default_rng(i) # Different seed for each restart
            local_calls = 0
            max_local_calls = budget_per_restart
            
            while local_calls < max_local_calls and calls_made < self.budget:
                if time.time() > deadline:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                try:
                    fy = call_func(y)
                except TimeoutError:
                    break
                
                local_calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                else:
                    sigma *= 0.7
                    if sigma < 1e-5:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
