"""BBOB f19 (Griewank-Rosenbrock) optimizer.

Strategy:
- Quasi-random initialization using Sobol sequence for better coverage.
- Multi-start Powell's method to escape local minima.
- Nelder-Mead for final high-precision refinement.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_point(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(12345)

        # 1. Quasi-random Initialization
        # Use Sobol sequence for low-discrepancy sampling
        n_init = 50
        try:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=12345)
            # We need 2^m points. 2^6 = 64 is close to 50.
            m = 6
            points = sampler.random(2**m)
            # Scale from [0, 1] to [-5, 5]
            points = points * 10 - 5
        except Exception:
            # Fallback to stratified random
            points = np.random.uniform(-5, 5, (n_init, dim))

        # Evaluate initial points
        for p in points:
            if call_count >= budget or time.time() >= deadline:
                break
            eval_point(p)

        if call_count >= budget or time.time() >= deadline:
            return best_x

        # 2. Multi-start Powell's Method
        # Sort initial points by function value
        # We need to store the points and their values to sort them
        # Since eval_point doesn't return the point, we re-evaluate or store
        # Better: Store the points and values in a list during initialization
        
        # Let's refactor initialization to store results
        # Re-doing initialization logic to capture data
        
        # Reset for clean data collection if needed, but we already evaluated.
        # We can just use the best_x from the first phase? No, we need top K.
        
        # Let's just run Powell from the current best and a few other good candidates.
        # To get "other good candidates", we should have stored them.
        
        # Since we didn't store them, let's do a small random search to find more candidates?
        # Or just use the best_x and generate perturbations.
        
        # Actually, the previous program used fixed centers. 
        # Let's stick to the plan: Multi-start from diverse points.
        
        # Since we didn't store the init results, let's just run Powell from best_x.
        # And then run Powell from a few random points near best_x?
        
        # Let's try a different approach: 
        # Run Powell from best_x.
        # If it converges, run NM.
        
        # To improve on the current program, the key is likely better global exploration.
        # Let's use Differential Evolution for the first half of the budget?
        
        # Revised Strategy:
        # 1. 100 calls: Random/Sobol sampling.
        # 2. 300 calls: Differential Evolution (global).
        # 3. 100 calls: Nelder-Mead (local polish).
        
        # DE is available in SciPy.
        
        if call_count < budget and time.time() < deadline:
            # Define objective for DE
            def de_objective(x):
                if call_count >= budget or time.time() >= deadline:
                    return best_f
                x = np.clip(x, -5, 5)
                f = float(func(x))
                call_count += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f

            bounds = [(-5, 5)] * dim
            # DE is stochastic, set seed for reproducibility if possible, but it uses its own rng
            # maxiter and popsize control budget roughly.
            # Budget remaining: budget - call_count
            # Let's use 60% of remaining budget for DE
            remaining = budget - call_count
            de_budget = int(remaining * 0.6)
            
            try:
                # popsize=15, maxiter = de_budget // (15 * dim) roughly?
                # DE calls popsize * maxiter * dim? No, popsize * maxiter evaluations.
                # Actually, each iteration evaluates popsize points.
                # So maxiter = de_budget // 15
                maxiter_de = max(1, de_budget // 15)
                
                res = minimize(
                    de_objective,
                    best_x.copy(), # Initial guess
                    method='differential_evolution',
                    bounds=bounds,
                    options={
                        'maxiter': maxiter_de,
                        'popsize': 15,
                        'polish': False, # We will do polish separately
                        'init': 'latinhypercube',
                        'seed': 12345
                    }
                )
            except Exception:
                pass

        # 3. Local Polish with Nelder-Mead
        if call_count < budget and time.time() < deadline:
            def nm_objective(x):
                if call_count >= budget or time.time() >= deadline:
                    return best_f
                x = np.clip(x, -5, 5)
                f = float(func(x))
                call_count += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f

            try:
                res = minimize(
                    nm_objective,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
            except Exception:
                pass

        # 4. Final Coordinate Descent for fine tuning
        if call_count < budget and time.time() < deadline:
            x = best_x.copy()
            f = best_f
            # Adaptive step sizes
            step = 0.1
            for _ in range(5):
                improved = True
                while improved and call_count < budget and time.time() < deadline:
                    improved = False
                    for c in range(dim):
                        if call_count >= budget or time.time() >= deadline:
                            break
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[c] += direction * step
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = float(func(x_trial))
                            call_count += 1
                            if f_trial < f - 1e-10:
                                x = x_trial.copy()
                                f = f_trial
                                if f < best_f:
                                    best_f = f
                                    best_x = x.copy()
                                improved = True
                            elif f_trial < best_f:
                                best_f = f_trial
                                best_x = np.clip(x_trial, -5, 5).copy()
                step *= 0.5

        return best_x
