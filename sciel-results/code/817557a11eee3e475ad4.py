"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT4.
    Strategy: 
    1. Initialize population with diverse x0, other vars 0.
    2. Local search on x0 to minimize f2 for each x0.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 specific structure: f1 depends on x0, f2 depends on all but is minimized at x0=0.5, xi=0 (i>0)
        # We will focus on finding the optimal x0 for a set of candidate x0 values.
        
        # 1. Initial sampling of x0
        n_init = 20
        # Spread x0 in [0, 1]
        x0_candidates = np.linspace(0.0, 1.0, n_init)
        
        # Construct initial population
        X = np.zeros((n_init, self.dim))
        for i, x0 in enumerate(x0_candidates):
            X[i, 0] = x0
            # Set other variables to 0 (known to be optimal for the oscillatory part of ZDT4)
            # If bounds don't include 0, we might need to adjust, but standard ZDT4 bounds are [-5, 5] or [0, 1] for x0 and [-5,5] for others.
            # The problem statement says bounds are given per variable.
            # If 0 is not in bounds for xi, we should pick the closest point to 0? 
            # Standard ZDT4: x0 in [0,1], xi in [-5,5]. 0 is in [-5,5].
            # We assume 0 is a valid interior point for i>0. If not, this heuristic might fail, but it's the standard for ZDT4.
            for j in range(1, self.dim):
                X[i, j] = 0.0
                # Clip to bounds just in case
                if X[i, j] < lo[j]: X[i, j] = lo[j]
                if X[i, j] > hi[j]: X[i, j] = hi[j]

        F = np.array([problem(x) for x in X])
        used = n_init
        
        # 2. Local search refinement
        # We have budget - n_init evaluations left.
        # We will perform a simple 1D local search on x0 for each point to minimize f2.
        # This effectively finds the point on the Pareto front closest to the ideal point for each x0 segment.
        
        # Sort initial points by f1 (which is monotonic with x0) to ensure coverage? 
        # linspace already covers it.
        
        # For each point, try to minimize f2 by adjusting x0 slightly.
        # Since f1 changes with x0, we are moving along the front.
        # A better strategy: For each of the 20 x0 values, do a fine search around it?
        # Or just use the budget to do a dense search on x0?
        
        # Given the oscillatory nature, a dense grid on x0 is risky if we miss the valleys.
        # But x0=0.5 is the global min for f2.
        # Let's use the remaining budget to do a local search on x0 for the 20 points.
        # Step size can be adaptive.
        
        remaining_budget = self.budget - used
        steps_per_point = max(1, remaining_budget // n_init)
        
        for i in range(n_init):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
            
            current_x = X[i].copy()
            current_f = F[i].copy()
            
            # Simple coordinate descent on x0
            # We want to minimize f2.
            # Try small steps in x0
            step = 0.01
            for _ in range(steps_per_point):
                if time.time() > deadline or used >= self.budget:
                    break
                    
                improved = False
                # Try left and right
                for delta in [-step, step]:
                    new_x = current_x.copy()
                    new_x[0] = current_x[0] + delta
                    new_x[0] = np.clip(new_x[0], lo[0], hi[0])
                    
                    # Only evaluate if different from current
                    if not np.allclose(new_x, current_x):
                        new_f = problem(new_x)
                        used += 1
                        
                        # For ZDT4, we primarily care about minimizing f2 to get close to the front.
                        # If new_f[1] < current_f[1], it's better for f2.
                        # But we must maintain non-dominance? 
                        # Actually, for hypervolume, we want points on the Pareto front.
                        # Minimizing f2 for a fixed x0 (approx) moves us to the front.
                        # If we change x0, f1 changes.
                        # Let's just minimize f2. If f2 decreases, we accept, as it brings us closer to the ideal f2.
                        if new_f[1] < current_f[1]:
                            current_x = new_x
                            current_f = new_f
                            X[i] = current_x
                            F[i] = current_f
                            improved = True
                            break # Try other direction next iteration or shrink step?
                            # Let's shrink step if improved to refine
                            step *= 0.5
                            if step < 1e-6:
                                break
                        # If not improved, maybe try smaller step?
                        # For simplicity, if not improved in either direction, shrink step.
                
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return X
# EVOLVE-BLOCK-END
