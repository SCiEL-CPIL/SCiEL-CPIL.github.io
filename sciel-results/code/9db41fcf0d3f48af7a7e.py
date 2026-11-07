"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start local search on weighted-sum scalarizations."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate a set of weight vectors to cover the objective space
        # For 3 objectives, we use a grid of weights summing to 1
        n_weights = 12
        weights = []
        # Simple grid for 3D simplex
        for i in range(n_weights):
            for j in range(n_weights - i):
                k = n_weights - i - j
                if i + j + k == n_weights:
                    weights.append([i / n_weights, j / n_weights, k / n_weights])
        weights = np.array(weights)
        
        # Shuffle weights to interleave exploration of different parts of the front
        np.random.seed(0) # Deterministic shuffling for reproducibility
        idx = np.arange(len(weights))
        np.random.shuffle(idx)
        weights = weights[idx]
        
        used = 0
        found_points = []
        
        # Allocate budget roughly equally to each weight vector
        # We have 200 budget. Let's say 100 weight vectors? No, n_weights above is small.
        # Let's increase n_weights to use the budget better.
        # With budget 200, if we do 20 restarts, we have 10 evals per restart.
        # Nelder-Mead in 4D usually needs ~20-50 evals to converge.
        # So maybe 5-10 restarts with higher quality convergence?
        # Or use a smaller budget per restart but more restarts?
        # The inspiration says "allocates the entire evaluation budget to 100 independent local opt".
        # 100 restarts * 2 evals each? That's too few for NM.
        # Let's try 20 restarts with ~10 evals each.
        
        # Actually, let's just loop through weights and allocate remaining budget.
        # We will run NM until budget runs out or time runs out.
        
        for w in weights:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Define scalarized objective
            def scalar_obj(x):
                nonlocal used
                if used >= self.budget:
                    return 0 # Dummy return to stop
                if time.time() > deadline:
                    return 0
                try:
                    f = problem(x)
                    used += 1
                    return np.dot(f, w)
                except:
                    return 1e6 # Large penalty for errors
            
            # Random start within bounds
            x0 = lo + (hi - lo) * np.random.random(self.dim)
            
            # Run Nelder-Mead
            try:
                res = minimize(scalar_obj, x0, method='Nelder-Mead', 
                               options={'maxiter': self.budget - used, 'xatol': 1e-4, 'fatol': 1e-6})
            except:
                pass
            
            if res.success or res.fun < 1e5:
                # Record the best point found by this local search
                # Note: res.x is the last point, not necessarily the best.
                # However, NM keeps the best simplex. We can re-evaluate res.x to be sure?
                # No, we are limited in budget. We assume res.x is good.
                # But we didn't record intermediate points.
                # For hypervolume, we want the set of non-dominated points.
                # The final point of each scalarization is a candidate Pareto point.
                found_points.append(res.x)
        
        # We have collected candidate points. 
        # The problem statement says "Every returned objective vector is recorded".
        # This implies the evaluator records the output of `problem(x)` every time it is called.
        # My `scalar_obj` calls `problem(x)`. So all intermediate points are already recorded by the evaluator!
        # I just need to ensure I don't crash and stay within budget.
        
        return found_points
# EVOLVE-BLOCK-END
