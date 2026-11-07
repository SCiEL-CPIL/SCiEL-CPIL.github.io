"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective Optimizer using Weighted Sum Scalarization with L-BFGS-B."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Strategy:
        # 1. Generate a set of diverse weight vectors to cover the Pareto front.
        # 2. For each weight vector, perform a local optimization (L-BFGS-B) starting from a random point.
        # 3. Record all evaluated points.
        
        # Budget allocation:
        # We don't know the cost of L-BFGS-B exactly, but we can monitor it.
        # Each L-BFGS-B call might use 10-100 evaluations depending on convergence.
        # With budget=200, we can afford roughly 5-20 local optimizations.
        
        # Let's generate weight vectors.
        # For 3 objectives, we can use a grid or random points on the simplex.
        # Simplex vertices: (1,0,0), (0,1,0), (0,0,1)
        # Edge midpoints: (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
        # Center: (1/3, 1/3, 1/3)
        # Plus some random weights to ensure coverage.
        
        # We will create a list of weight vectors.
        # Let's use a set of 10-15 weight vectors.
        
        n_weights = 12
        weights = []
        
        # Vertices
        weights.append([1.0, 0.0, 0.0])
        weights.append([0.0, 1.0, 0.0])
        weights.append([0.0, 0.0, 1.0])
        
        # Edge midpoints
        weights.append([0.5, 0.5, 0.0])
        weights.append([0.5, 0.0, 0.5])
        weights.append([0.0, 0.5, 0.5])
        
        # Center
        weights.append([1.0/3, 1.0/3, 1.0/3])
        
        # Random weights to fill the rest
        while len(weights) < n_weights:
            w = np.random.rand(n_obj)
            w = w / np.sum(w)
            weights.append(w.tolist())
        
        weights = np.array(weights)
        
        # We will iterate through these weights.
        # For each weight, we pick a random starting point within bounds.
        # We run L-BFGS-B with a max_iter limit to prevent hanging.
        
        # To maximize coverage, we should also do some pure random sampling?
        # L-BFGS-B converges to a local optimum. If the function is multi-modal, 
        # we need multiple starts per weight?
        # Given budget 200, if each L-BFGS-B takes ~10 evals, we can do ~20 optimizations.
        # Let's try 1 weight per optimization, but maybe restart if it fails?
        # Or just 1 start per weight.
        
        # Let's monitor evaluations carefully.
        
        eval_count = 0
        max_evals_per_opt = 20  # Limit per L-BFGS-B call
        
        # We'll store all evaluated points to ensure we don't lose any good ones
        # although the task says "Every returned objective vector is recorded", 
        # meaning the evaluator does this for us. We just need to call problem().
        
        # However, we should ensure we don't exceed budget.
        
        for w in weights:
            if time.time() > deadline:
                break
            if eval_count >= self.budget:
                break
            
            # Choose a random starting point
            x0 = lo + (hi - lo) * np.random.rand(dim)
            
            # Define scalar objective
            def scalar_obj(x, w=w):
                nonlocal eval_count
                if time.time() > deadline:
                    raise Exception("Time limit reached")
                if eval_count >= self.budget:
                    raise Exception("Budget exceeded")
                
                f = problem(x)
                eval_count += 1
                return np.dot(f, w)
            
            # Run L-BFGS-B
            # We use bounds
            bnds = [(lo[i], hi[i]) for i in range(dim)]
            
            try:
                res = minimize(scalar_obj, x0, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 50, 'maxfun': max_evals_per_opt})
            except Exception:
                # If it fails, just continue to next weight
                continue
            
            # Note: The evaluator records all calls to problem().
            # We don't need to do anything with res unless we want to use it for something else.
        
        return None
# EVOLVE-BLOCK-END
