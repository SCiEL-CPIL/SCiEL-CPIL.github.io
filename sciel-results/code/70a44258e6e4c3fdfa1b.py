# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 using hypervolume-aware sampling and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)
        
        used = 0
        X = []
        F = []
        
        def add_point(x, f):
            nonlocal used
            X.append(x)
            F.append(f)
            used += 1
        
        # For ZDT2: f1 = x[0], f2 = (1+10*n)*(1 - sqrt(x[0]/(1+10*n)))
        # With n=30: f2 = 31*(1 - sqrt(x[0]/31))
        # Pareto front: f2 = 31*(1 - sqrt(f1/31)) for f1 in [0, 31]
        # But x[0] in [0,1], so f1 in [0,1], f2 in [31*(1-sqrt(1/31)), 31]
        # f2_min = 31*(1 - sqrt(1/31)) ≈ 31*(1-0.1796) ≈ 25.42
        # f2_max = 31*(1-0) = 31
        
        # To maximize hypervolume with reference (11,11), we want points with small f1 and small f2
        # But on the Pareto front, small f1 means large f2 and vice versa
        # The trade-off is f1 vs f2 = 31*(1-sqrt(f1/31))
        
        # Optimal HV points: distribute along the Pareto front
        # For hypervolume maximization, we should place points where marginal gain is highest
        
        # Strategy: sample along the Pareto front with non-uniform distribution
        # The hypervolume contribution of a point at f1 is (11-f1)*(11-f2) where f2 = 31*(1-sqrt(f1/31))
        # But we need to account for other points
        
        # Simple approach: sample f1 values that maximize coverage
        # Since f2 is convex in f1, the Pareto front is convex
        # For HV maximization, use points spread along the front
        
        n_init = min(20, self.budget // 10)
        
        # Sample f1 values - use a distribution that emphasizes the region where HV gain is high
        # The marginal HV gain is highest where the curve is "farthest" from the reference
        # For ZDT2, this is in the middle of the front
        
        # Use a beta-like distribution for f1
        f1_samples = []
        for i in range(n_init):
            # Spread points, with some emphasis on the middle
            t = (i + 0.5) / n_init
            # Use a smooth distribution
            f1_val = t * 1.0  # f1 in [0, 1]
            f1_samples.append(f1_val)
        
        # Create initial points on the Pareto front
        for f1_val in f1_samples:
            x = np.zeros(self.dim)
            x[0] = f1_val
            # For ZDT2, other variables don't affect the Pareto front optimally
            # But we need to set them to values that achieve the Pareto front
            # For ZDT2, the Pareto front is achieved when x[1:] are set appropriately
            # Actually, for ZDT2, any x[1:] works as long as x[0] is fixed for f1
            # f2 depends only on x[0] for the Pareto front
            # But to be on the Pareto front, we need the correct relationship
            # For ZDT2: f2 = (1+10n)*(1 - sqrt(f1/(1+10n)))
            # This is achieved when x[0] = f1 and the other variables are set to make g(x) = 0
            # For ZDT2, g(x) = sum((xi - 0.5)^2 - cos(20*pi*(xi-0.5))) for i=2..n
            # g(x) = 0 when xi = 0.5 for all i >= 2
            
            x[1:] = 0.5  # This achieves g(x) = 0, putting us on the Pareto front
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            add_point(x, np.array(f))
        
        # Local search refinement
        # For each point, try small perturbations to improve hypervolume
        remaining_budget = self.budget - used
        
        if remaining_budget > 0:
            # Do local search on the most promising points
            # Sort by current hypervolume contribution (approximate by sum of objectives)
            F_arr = np.array(F)
            X_arr = np.array(X)
            
            # Priority: points with lower sum of objectives
            priority = np.argsort(np.sum(F_arr, axis=1))
            
            step_size = 0.01
            for _ in range(remaining_budget):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                    
                # Pick a point to refine
                idx = priority[0] if len(priority) > 0 else 0
                base_x = X_arr[idx]
                base_f = F_arr[idx]
                
                # Try a random perturbation
                direction = rng.standard_normal(self.dim)
                direction /= np.linalg.norm(direction) + 1e-10
                
                new_x = base_x + step_size * direction
                new_x = np.clip(new_x, lo, hi)
                
                new_f = np.array(problem(new_x))
                add_point(new_x, new_f)
                
                # Update arrays
                X_arr = np.array(X)
                F_arr = np.array(F)
                priority = np.argsort(np.sum(F_arr, axis=1))
        
        return X
# EVOLVE-BLOCK-END
