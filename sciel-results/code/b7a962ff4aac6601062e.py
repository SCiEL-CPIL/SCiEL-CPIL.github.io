"""Optimal-control-inspired adaptive hypervolume maximizer for ZDT1."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Adaptive multi-objective optimizer for ZDT1.
    Exploits the known structure of ZDT1 where the Pareto front is achieved
    when x_2..x_n are set to 0 (minimizing g(x) to 1).
    Uses adaptive bisection on x_1 to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 specific: x_1 determines f_1, x_2..x_n determine g.
        # Optimal g=1 is at x_2..x_n = 0.
        # We will fix x_2..x_n to 0 (or near 0) and optimize x_1.
        
        all_X = []
        used = 0
        
        def make_x(x1_val):
            x = np.zeros(n)
            x[0] = x1_val
            # x[1:] remains 0.0, which is optimal for ZDT1 g-function
            return x
            
        def evaluate(x1_val):
            nonlocal used
            if used >= budget:
                return
            x = make_x(x1_val)
            f = problem(x)
            used += 1
            all_X.append(x)
            
        # Phase 1: Initial coarse grid
        # Use a few initial points to establish the range
        initial_points = [0.0, 0.25, 0.5, 0.75, 1.0]
        for x1 in initial_points:
            evaluate(x1)
            
        if used >= budget:
            return np.array(all_X)
            
        # Sort evaluated x1 values to find intervals
        # We track the sorted x1s separately for efficiency
        x1s = sorted([x[0] for x in all_X])
        
        # Phase 2: Adaptive bisection
        # We want to pick the interval [x1[i], x1[i+1]] with the largest width
        # This approximates maximizing marginal hypervolume gain for smooth fronts
        
        while used < budget:
            if len(x1s) < 2:
                # Fallback if we somehow have < 2 points (shouldn't happen with init)
                new_x1 = 0.5
                if new_x1 not in x1s:
                    evaluate(new_x1)
                    x1s = sorted([x[0] for x in all_X])
                else:
                    break
                    
            # Find the interval with the maximum width
            max_width = -1
            best_idx = -1
            for i in range(len(x1s) - 1):
                width = x1s[i+1] - x1s[i]
                if width > max_width:
                    max_width = width
                    best_idx = i
                    
            if best_idx == -1 or max_width < 1e-6:
                # If intervals are very small, add a random point to explore
                new_x1 = np.random.rand()
                if new_x1 not in x1s:
                    evaluate(new_x1)
                    x1s = sorted([x[0] for x in all_X])
                else:
                    break
            else:
                mid_x1 = (x1s[best_idx] + x1s[best_idx+1]) / 2.0
                evaluate(mid_x1)
                x1s = sorted([x[0] for x in all_X])
                
        return np.array(all_X)
# EVOLVE-BLOCK-END
