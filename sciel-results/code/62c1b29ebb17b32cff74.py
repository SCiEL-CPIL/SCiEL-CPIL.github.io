"""Rewritten optimizer for RE21 using Sobol initialization and multi-start L-BFGS-B refinement."""
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = None
        self.eval_count = 0
        self.problem = None
        self.bounds = None
        self.lo = None
        self.hi = None
        self.results = []

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = self.start_time + 290  # Safety margin for 330s limit
        self.eval_count = 0
        self.problem = problem
        self.bounds = bounds
        self.lo = np.array([b[0] for b in bounds])
        self.hi = np.array([b[1] for b in bounds])
        self.results = []
        
        # 1. Global Exploration with Sobol Sequence
        # Use 30% of budget for initial sampling
        n_init = max(10, int(self.budget * 0.3))
        n_init = min(n_init, self.budget)
        
        # Generate Sobol points
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        try:
            points = sobol.random(n_init)
        except Exception:
            points = np.random.rand(n_init, self.dim)
            
        X_init = self.lo + points * (self.hi - self.lo)
        F_init = np.array([self._evaluate(x) for x in X_init])
        
        self.results.extend(zip(X_init, F_init))
        
        if self.eval_count >= self.budget:
            return np.array([r[0] for r in self.results])
        
        # 2. Multi-start L-BFGS-B with weighted-sum scalarization
        # Use remaining budget for local optimization
        # Each L-BFGS-B call can use up to ~20 evaluations (maxiter=20)
        # We'll do multiple starts with different weights
        
        # Define weight combinations for 2 objectives
        # We want to explore the Pareto front, so use various weight combinations
        n_weights = 8  # Number of different weight combinations
        weights = np.linspace(0.05, 0.95, n_weights)
        
        # For each weight, pick the best starting point from initial sampling
        # based on the scalarized objective
        for w in weights:
            if self.eval_count >= self.budget:
                break
            if time.time() > deadline:
                break
                
            # Scalarized objective: w*f1 + (1-w)*f2
            def scalarized_obj(x):
                f = self._evaluate(x)
                return w * f[0] + (1 - w) * f[1]
            
            # Find best starting point from initial results
            best_idx = 0
            best_val = float('inf')
            for i, (xi, fi) in enumerate(self.results):
                val = w * fi[0] + (1 - w) * fi[1]
                if val < best_val:
                    best_val = val
                    best_idx = i
            
            x0 = self.results[best_idx][0].copy()
            
            # Run L-BFGS-B
            try:
                res = minimize(scalarized_obj, x0, method='L-BFGS-B', 
                              bounds=self.bounds, options={'maxiter': 15, 'maxfun': 15})
                if res.success and self.eval_count < self.budget:
                    self.results.append((res.x, self._evaluate(res.x)))
            except Exception:
                pass
        
        # 3. Additional random restarts if budget remains
        if self.eval_count < self.budget:
            n_extra = min(5, self.budget - self.eval_count)
            for _ in range(n_extra):
                if self.eval_count >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Random point
                x_rand = self.lo + np.random.rand(self.dim) * (self.hi - self.lo)
                # Use a random weight
                w = np.random.rand()
                
                def scalarized_obj(x):
                    f = self._evaluate(x)
                    return w * f[0] + (1 - w) * f[1]
                
                try:
                    res = minimize(scalarized_obj, x_rand, method='L-BFGS-B', 
                                  bounds=self.bounds, options={'maxiter': 10, 'maxfun': 10})
                    if res.success and self.eval_count < self.budget:
                        self.results.append((res.x, self._evaluate(res.x)))
                except Exception:
                    pass
        
        # Return all evaluated points
        X_all = np.array([r[0] for r in self.results])
        return X_all
    
    def _evaluate(self, x):
        """Evaluate with budget tracking and time check."""
        if time.time() > self.start_time + 295:
            raise Exception("Time limit exceeded")
        
        if self.eval_count >= self.budget:
            raise Exception("Budget exceeded")
        
        self.eval_count += 1
        f = self.problem(x)
        return np.array(f)

# EVOLVE-BLOCK-END
