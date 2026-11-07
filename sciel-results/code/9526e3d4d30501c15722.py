import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 325.0  # 5s safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Phase 1: Global Search using Latin Hypercube Sampling (LHS)
        # BBOB-21 has many local optima. We need a dense global search to locate
        # the basin of the global optimum. We allocate 60% of the budget here.
        n_global = int(budget * 0.6)
        n_global = max(n_global, 50)
        
        # Generate LHS points in [-5, 5]^dim
        lhs = np.zeros((n_global, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n_global + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            lhs[:, d] = points

        # Evaluate global points
        f_values = np.zeros(n_global)
        calls = 0
        
        for i in range(n_global):
            if time.time() > deadline:
                break
            f_values[i] = func(lhs[i])
            calls += 1
            
        if calls == 0:
            return np.zeros(dim), 0.0

        # Sort by fitness
        sorted_indices = np.argsort(f_values)
        best_x = lhs[sorted_indices[0]].copy()
        best_f = f_values[sorted_indices[0]]

        # Phase 2: Local Refinement using (1+1)-ES
        # We take the top K candidates from the global search and run (1+1)-ES on them.
        # (1+1)-ES is robust and efficient for the final convergence.
        
        n_local_candidates = 4
        remaining_budget = budget - calls
        
        if remaining_budget <= 0:
            return best_x, best_f
        
        # Allocate budget per local search
        # With 60% global, we have 40% local. 
        # 200 calls / 4 candidates = 50 calls each.
        budget_per_local = max(10, remaining_budget // n_local_candidates)
        
        for k in range(n_local_candidates):
            if calls >= budget or time.time() > deadline:
                break
            
            # Start from the k-th best global sample
            x = lhs[sorted_indices[k]].copy()
            f = f_values[sorted_indices[k]]
            
            # (1+1)-ES parameters
            # Initial step size: 
            # If we are close to a boundary, maybe smaller? 
            # Generally 0.5 is a good start for [-5,5].
            sigma = 0.5
            n_evals_local = 0
            max_local_evals = budget_per_local
            
            # 1/5th rule parameters
            success_rate_target = 0.2
            sigma_increase = 1.1
            sigma_decrease = 0.9
            window_size = 10
            window = []
            
            while n_evals_local < max_local_evals and calls < budget and time.time() < deadline:
                # Generate mutant
                mutation = rng.normal(0, sigma, size=dim)
                x_new = x + mutation
                x_new = np.clip(x_new, -5, 5)
                
                # Evaluate mutant
                f_new = func(x_new)
                calls += 1
                n_evals_local += 1
                
                # Update window for 1/5th rule
                if f_new < f:
                    window.append(1)
                    # Accept
                    x = x_new
                    f = f_new
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    window.append(0)
                
                if len(window) >= window_size:
                    success_rate = np.mean(window[-window_size:])
                    if success_rate > success_rate_target:
                        sigma *= sigma_increase
                    elif success_rate < success_rate_target:
                        sigma *= sigma_decrease
                    # Clamp sigma
                    sigma = np.clip(sigma, 1e-6, 10.0)
                
                # Check for convergence
                if sigma < 1e-8:
                    break

        return best_x, best_f
