import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # WFG9: Pareto front controlled primarily by x[0]
        # Use dense sweep along x[0] with remaining vars at midpoint (or optimized)
        n_sweep = 30
        sweep_vals = np.linspace(0.01, 0.99, n_sweep)
        
        # Initial population: sweep along x[0], other vars at center
        X = []
        F = []
        for t in sweep_vals:
            x = np.full(self.dim, 0.5)  # center for non-trade-off vars
            x[0] = t
            # Scale to bounds
            x_scaled = lo + span * x
            X.append(x_scaled)
            F.append(problem(x_scaled))
        
        X = np.array(X)
        F = np.array(F)
        used = n_sweep
        
        # Local refinement: for each point, perturb non-trade-off vars to minimize distance
        # Use remaining budget for local search
        remaining_budget = self.budget - used
        n_local = min(remaining_budget, 50)
        
        # For each sweep point, do local refinement on non-trade-off variables
        points_per_refine = max(1, n_local // n_sweep)
        
        for i in range(n_sweep):
            if used >= self.budget:
                break
            if time.time() >= deadline:
                break
            
            # Local search around X[i] for non-trade-off variables
            best_x = X[i].copy()
            best_f = F[i].copy()
            
            for _ in range(points_per_refine):
                if used >= self.budget:
                    break
                if time.time() >= deadline:
                    break
                
                # Perturb non-trade-off variables (indices 1+)
                perturbation = 0.05 * span
                child = best_x.copy()
                child[1:] += perturbation * rng.standard_normal(self.dim - 1)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Accept if it improves (dominates or reduces sum)
                if np.all(fc <= best_f) or np.sum(fc) < np.sum(best_f):
                    best_x = child
                    best_f = fc
            
            X[i] = best_x
            F[i] = best_f
        
        # Final: keep all non-dominated solutions
        dominated = np.array([np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F])
        non_dom_mask = ~dominated
        
        # Return the non-dominated solutions
        return X[non_dom_mask]
