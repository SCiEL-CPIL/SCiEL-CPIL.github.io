import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        
        best_x = None
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)
        
        def wrapped_func(x):
            nonlocal best_x, best_f, call_count
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if call_count >= self.budget:
                raise StopIteration("Budget exceeded")
            
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            return fx

        # Phase 1: Latin Hypercube Sampling for exploration
        # Use ~80 LHS points in [-4, 4]^dim to quickly locate the basin
        n_lhs = min(80, self.budget // 6)
        lhs_points = []
        for i in range(n_lhs):
            x = np.zeros(self.dim)
            for d in range(self.dim):
                stratum = (i + rng.random()) / n_lhs
                x[d] = -4.0 + 8.0 * stratum
            lhs_points.append(x)
        
        rng.shuffle(lhs_points)
        
        for x in lhs_points:
            if time.time() > deadline or call_count >= self.budget:
                break
            try:
                wrapped_func(x)
            except (TimeoutError, StopIteration):
                break
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            call_count = 1
        
        # Phase 2: (1+1)-ES refinement from the best point
        # Adaptive step size, no restarts from perturbed points
        remaining_budget = self.budget - call_count
        if remaining_budget > 0 and best_x is not None:
            local_best_x = best_x.copy()
            local_best_f = best_f
            local_sigma = 1.0  # Start with reasonable step size
            
            # (1+1)-ES with adaptive step size
            no_improve = 0
            for step in range(min(remaining_budget, 350)):
                if call_count >= self.budget or time.time() > deadline:
                    break
                
                y = np.clip(local_best_x + local_sigma * rng.standard_normal(self.dim), -5.0, 5.0)
                fy = wrapped_func(y)
                
                if fy < local_best_f:
                    local_best_x = y.copy()
                    local_best_f = fy
                    no_improve = 0
                    # Increase step size on success (but cap it)
                    local_sigma = min(local_sigma * 1.1, 3.0)
                else:
                    no_improve += 1
                    # Decrease step size on failure
                    local_sigma *= 0.9
                    
                    # If stuck for many iterations, try a larger jump
                    if no_improve > 20:
                        local_sigma *= 2.0
                        no_improve = 0
                
                # Update global best
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()
        
        # Phase 3: Final polish with very small steps
        remaining_budget = self.budget - call_count
        if remaining_budget > 5 and best_x is not None:
            local_best_x = best_x.copy()
            local_best_f = best_f
            local_sigma = 0.05
            
            for step in range(min(remaining_budget, 100)):
                if call_count >= self.budget or time.time() > deadline:
                    break
                
                y = np.clip(local_best_x + local_sigma * rng.standard_normal(self.dim), -5.0, 5.0)
                fy = wrapped_func(y)
                
                if fy < local_best_f:
                    local_best_x = y.copy()
                    local_best_f = fy
                    # Slowly decrease step size for precision
                    local_sigma *= 0.95
                else:
                    local_sigma *= 0.9
                
                # Update global best
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()
        
        return best_x, best_f
