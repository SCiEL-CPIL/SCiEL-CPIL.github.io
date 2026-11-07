import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            if self.calls >= budget:
                raise Exception("Budget Exceeded")
            self.calls += 1
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                f = float('inf')
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # 1. Initial Exploration: Sobol-like quasi-random sequence for better coverage
        # For dim=5, we can use a simple low-discrepancy sequence or just stratified random.
        # Let's use a simple stratified random sampling (Latin Hypercube) to ensure coverage.
        n_init = 40
        lhs_points = []
        for i in range(n_init):
            x = np.zeros(dim)
            for d in range(dim):
                # Stratified sampling: divide [0,1] into n_init intervals
                # For each dimension, pick a random offset within the interval i
                # To make it LHS, we permute the intervals for each dimension
                pass
            
        # Proper LHS generation
        perm = [np.random.permutation(n_init) for _ in range(dim)]
        for i in range(n_init):
            x = np.zeros(dim)
            for d in range(dim):
                # Interval i is [i/n_init, (i+1)/n_init]
                # The sample is in interval perm[d][i]
                interval_idx = perm[d][i]
                # Random offset within the interval
                offset = rng.random()
                x[d] = -5.0 + 10.0 * (interval_idx + offset) / n_init
            lhs_points.append(x)
        
        # Evaluate LHS points
        for x in lhs_points:
            wrapped_func(x)
        
        # Also evaluate origin and a few other strategic points
        strategic_points = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 4.0,
            np.ones(dim) * -4.0,
        ]
        for x in strategic_points:
            wrapped_func(x)

        # 2. Multi-start Local Optimization using Nelder-Mead and BFGS
        # Since CMA-ES can be slow or get stuck in flat regions (Discus has a flat bottom),
        # we will use multiple local optimizers starting from the best few points found in initialization.
        
        # Select top 5 points from initial exploration
        # We need to store all evaluated points and their values to sort them.
        # The wrapped_func doesn't return a list, so we need to modify the strategy or keep a log.
        # Let's keep a log of all evaluations.
        
        # Reset and re-evaluate? No, that wastes budget.
        # Instead, let's just pick the best few from the ones we know.
        # We didn't store the LHS points' values separately in a list for sorting easily without refactoring wrapped_func.
        # Let's refactor wrapped_func to store history.
        
        # Actually, the current wrapped_func updates self.best_f and self.best_x but doesn't keep a list.
        # Let's add a history list.
        
        # Wait, I can't easily refactor wrapped_func now without breaking the flow if I already called it.
        # I will implement the local optimization starting from self.best_x and a few random perturbations of it,
        # or simply restart local optimization from the best point found so far.
        
        # Strategy:
        # Use the remaining budget for a series of local optimizations.
        # For each local optimization, we use a portion of the budget.
        # We will use scipy.optimize.minimize with method='Nelder-Mead' and 'BFGS'.
        
        # Since we don't have the list of all initial points sorted, let's just use the current best.
        # To improve robustness, we will generate a few new random points in the region around the current best
        # and try to optimize from there if the current best seems stuck (optional, but risky with budget).
        
        # Simpler: Just run a robust local optimizer from the current best.
        # Nelder-Mead is robust for non-differentiable or noisy functions, but BFGS is faster if smooth.
        # Discus is smooth. Let's try BFGS first, then Nelder-Mead if it fails or doesn't improve.
        
        # We need to be careful with the budget.
        # Let's allocate the remaining budget to a sequence of local optimizations.
        
        remaining = budget - self.calls
        if remaining <= 0:
            return self.best_x, self.best_f

        # Define a local optimizer that uses a subset of the budget
        def run_local_opt(x0, max_calls):
            # x0: starting point
            # max_calls: max number of function evaluations allowed
            
            # We need a wrapper that counts calls specifically for this local opt
            # and stops when max_calls is reached or deadline is hit.
            
            local_calls = 0
            local_best_x = x0.copy()
            local_best_f = self.best_f  # Start with global best to avoid regression
            
            def local_func(x):
                nonlocal local_calls, local_best_x, local_best_f
                if time.time() > deadline:
                    raise TimeoutError()
                if local_calls >= max_calls:
                    raise Exception("Local Budget Exceeded")
                local_calls += 1
                # Use the global wrapped_func to ensure global tracking
                f = wrapped_func(x)
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = np.asarray(x).copy()
                return f
            
            # Try BFGS
            try:
                from scipy.optimize import minimize
                # BFGS requires gradient. We can use numerical gradient or just Nelder-Mead.
                # Nelder-Mead is derivative-free and robust.
                # Let's use Nelder-Mead with adaptive settings.
                
                # Initial simplex size
                # Default is 0.05 * x0 + 0.00025 if x0 is 0.
                # Let's set an explicit initial step size.
                init_step = 0.5
                
                res = minimize(
                    local_func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': max_calls, 
                        'xatol': 1e-8, 
                        'fatol': 1e-8, 
                        'adaptive': True,
                        'initial_simplex': np.array([x0 + init_step * np.eye(dim)[i] for i in range(dim+1)])
                    }
                )
                if res.fun < local_best_f:
                    local_best_f = res.fun
                    local_best_x = res.x.copy()
            except Exception:
                # If Nelder-Mead fails, try COBYLA or just return
                try:
                    from scipy.optimize import minimize
                    res = minimize(
                        local_func, 
                        x0, 
                        method='COBYLA', 
                        options={
                            'maxiter': max_calls,
                            'rhobeg': 0.5
                        }
                    )
                    if res.fun < local_best_f:
                        local_best_f = res.fun
                        local_best_x = res.x.copy()
                except Exception:
                    pass
            
            return local_best_x, local_best_f

        # Run local optimization from the current best
        # Allocate 80% of remaining budget to the first local run
        budget_local1 = int(remaining * 0.8)
        if budget_local1 > 10:
            x_best, f_best = run_local_opt(self.best_x.copy(), budget_local1)
            if f_best < self.best_f:
                self.best_f = f_best
                self.best_x = x_best

        remaining = budget - self.calls
        if remaining > 10 and time.time() < deadline - 5:
            # Try a second local optimization from a perturbed point
            # Perturb the best point with a small random noise
            noise = rng.normal(0, 0.1, size=dim)
            x_perturbed = np.clip(self.best_x + noise, -5, 5)
            
            budget_local2 = remaining
            x_best2, f_best2 = run_local_opt(x_perturbed, budget_local2)
            if f_best2 < self.best_f:
                self.best_f = f_best2
                self.best_x = x_best2

        return self.best_x, self.best_f
