import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc
from scipy.interpolate import RBFInterpolator

class Optimizer:
    """
    Optimizer using Bayesian Optimization with RBF surrogate.
    1. Initial exploration with Latin Hypercube Sampling.
    2. Iteratively fit an RBF surrogate and use Expected Improvement to select next points.
    3. Periodic local polish with Nelder-Mead on the best point.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        total_budget = self.budget
        
        calls = 0
        best_x = np.zeros(n)
        best_f = np.inf
        
        # Data storage for surrogate
        X_data = []
        f_data = []
        
        # Helper to clip to box
        def clip(x):
            return np.clip(x, -5, 5)

        # Evaluate function wrapper
        def eval_func(x):
            nonlocal calls, best_f, best_x
            if calls >= total_budget or time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (n,):
                x = x.reshape(n)
            x = clip(x)
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            calls += 1
            if np.isfinite(fx):
                X_data.append(x.copy())
                f_data.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial Exploration
        n_explore = min(30, total_budget)
        
        if n_explore >= total_budget:
            points = np.random.uniform(-5, 5, (n_explore, n))
        else:
            sampler = qmc.LatinHypercube(d=n, seed=42)
            samples = sampler.random(n_explore)
            points = -5 + 10 * samples
        
        for p in points:
            if calls >= total_budget or time.time() > deadline:
                break
            eval_func(p)

        # Phase 2: Bayesian Optimization Loop
        # Fit RBF surrogate and use EI acquisition
        min_budget_per_loop = 5
        polish_interval = 40  # every 40 BO iterations, do a local polish
        
        iteration = 0
        while calls < total_budget and time.time() < deadline:
            iteration += 1
            
            # Check if we have enough data to fit surrogate
            if len(X_data) < 3:
                # Not enough data, add random points
                n_add = min(10, total_budget - calls)
                for _ in range(n_add):
                    if calls >= total_budget or time.time() > deadline:
                        break
                    p = np.random.uniform(-5, 5, n)
                    eval_func(p)
                continue
            
            # Fit RBF surrogate
            X_arr = np.array(X_data)
            f_arr = np.array(f_data)
            
            try:
                # Use RBF interpolation as surrogate
                interpolator = RBFInterpolator(
                    X_arr, f_arr, 
                    kernel='thin_plate_spline',
                    smoothing=1e-3
                )
            except Exception:
                # Fallback to simpler kernel
                try:
                    interpolator = RBFInterpolator(
                        X_arr, f_arr, 
                        kernel='gaussian',
                        smoothing=1e-2
                    )
                except Exception:
                    # Can't fit, do random search
                    p = np.random.uniform(-5, 5, n)
                    eval_func(p)
                    continue
            
            # Expected Improvement acquisition
            # EI(x) = E[max(f_best - f(x), 0)]
            # For minimization: EI(x) = E[max(f_best - f(x), 0)]
            # We approximate by sampling candidate points and computing EI using the surrogate
            
            f_best = min(f_arr)
            
            # Generate candidate points for acquisition
            n_candidates = 200
            # Mix of random and perturbed best points
            candidates = np.random.uniform(-5, 5, (n_candidates // 2, n))
            
            # Perturb best point and other good points
            n_good = min(5, len(X_arr))
            good_indices = np.argsort(f_arr)[:n_good]
            for idx in good_indices:
                n_perturb = max(1, n_candidates // 2 // n_good)
                perturbations = np.random.normal(0, 0.5, (n_perturb, n))
                perturbed = X_arr[idx] + perturbations
                perturbed = np.clip(perturbed, -5, 5)
                candidates = np.vstack([candidates, perturbed])
            
            # Truncate to max candidates
            candidates = candidates[:n_candidates]
            
            # Evaluate surrogate on candidates
            try:
                f_pred = interpolator(candidates)
            except Exception:
                f_pred = np.random.normal(f_best, 1.0, len(candidates))
            
            # Compute Expected Improvement
            # EI = (f_best - f_pred) * Phi((f_best - f_pred)/sigma) + sigma * phi((f_best - f_pred)/sigma)
            # Since we don't have uncertainty from RBF, use a simple heuristic:
            # Prefer points where surrogate predicts low values, but also explore
            improvement = f_best - f_pred
            
            # Add small randomization to avoid getting stuck
            noise = np.random.normal(0, 0.01 * max(1, abs(f_best)), len(candidates))
            scores = improvement + noise
            
            # Select best candidate
            best_idx = np.argmax(scores)
            next_x = candidates[best_idx]
            
            # Evaluate the selected point
            eval_func(next_x)
            
            # Periodic local polish
            if iteration % polish_interval == 0 and calls < total_budget - 20:
                # Do a quick Nelder-Mead polish from best_x
                remaining = total_budget - calls
                if remaining > 20:
                    budget_polish = min(30, remaining - 5)
                    
                    def local_func(x):
                        if calls >= total_budget or time.time() > deadline:
                            raise ValueError("Budget Exceeded")
                        return eval_func(x)
                    
                    try:
                        res = minimize(local_func, best_x.copy(), method='Nelder-Mead',
                                       options={'maxiter': max(1, budget_polish // 5), 
                                                'xatol': 1e-8, 'fatol': 1e-8})
                    except Exception:
                        pass

        return best_x, best_f
