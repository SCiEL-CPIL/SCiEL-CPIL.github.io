"""Optimizer for BBOB-style f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s, keep 30s buffer
        deadline = start_time + 300.0 
        
        dim = self.dim
        total_budget = self.budget
        
        # State tracking
        self.calls = 0
        self.best_x = np.zeros(dim)
        self.best_f = np.inf
        self.deadline = deadline

        def wrapped_func(x):
            # Check budget and time
            if self.calls >= self.budget:
                return np.inf
            if time.time() > self.deadline:
                return np.inf
            
            x = np.clip(x, -5, 5)
            self.calls += 1
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # 1. Initial Diverse Sampling (Global Exploration)
        # We use a small fraction of the budget to find good basins.
        # f19 has a Griewank component which creates many local minima.
        # We want to hit at least one good basin early.
        
        init_budget = 20
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Axis-aligned points to probe the Griewank lobes
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = 4.0
            p2 = np.zeros(dim); p2[i] = -4.0
            starts.append(p1)
            starts.append(p2)
        # Random uniform
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, size=dim))
        # Random Gaussian
        for _ in range(5):
            starts.append(rng.normal(0, 1.5, size=dim))
        
        # Evaluate initial points
        # We prioritize evaluating the most likely good points first? 
        # Or just evaluate a few to get a baseline.
        # Let's evaluate the first 10-15 points to find a good start.
        
        evaluated_starts = []
        for x0 in starts:
            if self.calls >= init_budget:
                break
            f0 = wrapped_func(x0)
            if np.isfinite(f0):
                evaluated_starts.append((f0, x0.copy()))
        
        # Sort by function value
        evaluated_starts.sort(key=lambda x: x[0])
        
        # If we didn't find anything finite, fallback to origin
        if not evaluated_starts:
            evaluated_starts.append((self.best_f, self.best_x.copy()))

        # 2. Multi-start Nelder-Mead (Local Refinement)
        # We will run NM on the top 3 distinct starting points found in phase 1.
        # Also, we will run a final polish on the global best.
        
        # Determine unique starts (avoiding duplicates)
        unique_starts = []
        for f, x in evaluated_starts:
            is_dup = False
            for _, ux in unique_starts:
                if np.linalg.norm(x - ux) < 1e-5:
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append((f, x))
            
            if len(unique_starts) >= 3:
                break
        
        # If we have fewer than 3, add random ones
        while len(unique_starts) < 3 and self.calls < self.budget:
             new_x = rng.uniform(-4, 4, size=dim)
             f_new = wrapped_func(new_x)
             if np.isfinite(f_new):
                 unique_starts.append((f_new, new_x.copy()))
        
        # Allocate budget for NM runs
        # NM in 5D is fast. 1 iteration ~ 6-10 calls.
        # Total budget remaining: total_budget - self.calls
        remaining = total_budget - self.calls
        
        # Strategy: 
        # Run 1: Best start. Give it 40% of remaining.
        # Run 2: 2nd best start. Give it 30% of remaining.
        # Run 3: 3rd best start. Give it 20% of remaining.
        # Polish: Last 10% on global best.
        
        # To maximize AOCC, we want the first run to start immediately.
        
        def run_nm(x0, budget_frac):
            if self.calls >= self.budget:
                return
            if time.time() > self.deadline - 5:
                return
            
            max_iter = int(budget_frac * remaining / 8) # ~8 calls per iter
            max_iter = max(10, max_iter) # At least 10 iterations
            
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead',
                               options={
                                   'maxiter': max_iter,
                                   'xatol': 1e-6,
                                   'fatol': 1e-10,
                                   'adaptive': True
                               })
            except Exception:
                pass

        # Run 1: Best start
        if unique_starts:
            run_nm(unique_starts[0][1], 0.4)
        
        # Run 2: Second best start
        if len(unique_starts) > 1:
             run_nm(unique_starts[1][1], 0.3)
             
        # Run 3: Third best start
        if len(unique_starts) > 2:
             run_nm(unique_starts[2][1], 0.2)
             
        # Final Polish: High precision on the global best found so far
        if self.calls < self.budget and time.time() < self.deadline - 10:
            polish_budget = self.budget - self.calls
            max_iter_polish = max(10, int(polish_budget / 5))
            try:
                # Use a smaller initial simplex for polish
                simplex = [self.best_x.copy()]
                for i in range(dim):
                    v = self.best_x.copy()
                    v[i] += 0.01
                    simplex.append(v)
                
                res = minimize(wrapped_func, self.best_x, method='Nelder-Mead',
                               options={
                                   'maxiter': max_iter_polish,
                                   'xatol': 1e-8,
                                   'fatol': 1e-12,
                                   'adaptive': True,
                                   'initial_simplex': np.array(simplex)
                               })
            except Exception:
                pass

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
