import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from scipy.interpolate import RBFInterpolator

class Optimizer:
    """
    Optimizer using Gaussian Process surrogate with Expected Improvement.
    1. Initial exploration with Latin Hypercube Sampling.
    2. Fit GP surrogate, use EI acquisition for model-guided search.
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

        # Phase 2: Bayesian Optimization Loop with GP surrogate
        polish_interval = 25  # more frequent polish to exploit progress
        
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
            
            # Fit GP surrogate using RBF kernel with uncertainty
            X_arr = np.array(X_data)
            f_arr = np.array(f_data)
            
            # Normalize for better GP conditioning
            f_mean = np.mean(f_arr)
            f_std = np.std(f_arr) if np.std(f_arr) > 1e-10 else 1.0
            f_norm = (f_arr - f_mean) / f_std
            
            try:
                # Use RBF with smoothing to approximate GP mean and variance
                interpolator = RBFInterpolator(
                    X_arr, f_norm, 
                    kernel='gaussian',
                    smoothing=1e-4
                )
            except Exception:
                # Fallback to thin plate spline
                try:
                    interpolator = RBFInterpolator(
                        X_arr, f_norm, 
                        kernel='thin_plate_spline',
                        smoothing=1e-3
                    )
                except Exception:
                    # Can't fit, do random search
                    p = np.random.uniform(-5, 5, n)
                    eval_func(p)
                    continue
            
            f_best_norm = min(f_norm)
            
            # Generate candidate points for acquisition
            n_candidates = 150
            # Mix of random and perturbed best points
            candidates = np.random.uniform(-5, 5, (n_candidates // 2, n))
            
            # Perturb best point and other good points
            n_good = min(5, len(X_arr))
            good_indices = np.argsort(f_norm)[:n_good]
            for idx in good_indices:
                n_perturb = max(1, (n_candidates // 2) // n_good)
                perturbations = np.random.normal(0, 0.3, (n_perturb, n))
                perturbed = X_arr[idx] + perturbations
                perturbed = np.clip(perturbed, -5, 5)
                candidates = np.vstack([candidates, perturbed])
            
            # Truncate to max candidates
            candidates = candidates[:n_candidates]
            
            # Evaluate surrogate on candidates
            try:
                f_pred_norm = interpolator(candidates)
            except Exception:
                f_pred_norm = np.random.normal(f_best_norm, 1.0, len(candidates))
            
            # Compute Expected Improvement with uncertainty estimate
            # Approximate uncertainty based on distance to nearest data points
            dists = []
            for c in candidates:
                d = np.min(np.linalg.norm(X_arr - c, axis=1))
                dists.append(d)
            dists = np.array(dists)
            
            # Uncertainty increases with distance from data
            max_dist = np.max(dists) if np.max(dists) > 1e-10 else 1.0
            uncertainty = 0.5 * (dists / max_dist) + 0.1
            
            # EI for minimization: EI = (f_best - f_pred) * Phi((f_best - f_pred)/sigma) + sigma * phi((f_best - f_pred)/sigma)
            # Where Phi is CDF and phi is PDF of standard normal
            diff = f_best_norm - f_pred_norm
            z = diff / (uncertainty + 1e-10)
            ei = diff * norm.cdf(z) + uncertainty * norm.pdf(z)
            
            # Add small randomization
            noise = np.random.normal(0, 0.005, len(candidates))
            scores = ei + noise
            
            # Select best candidate
            best_idx = np.argmax(scores)
            next_x = candidates[best_idx]
            
            # Evaluate the selected point
            eval_func(next_x)
            
            # Periodic local polish
            if iteration % polish_interval == 0 and calls < total_budget - 15:
                # Do a quick Nelder-Mead polish from best_x
                remaining = total_budget - calls
                if remaining > 15:
                    budget_polish = min(25, remaining - 3)
                    
                    def local_func(x):
                        if calls >= total_budget or time.time() > deadline:
                            raise ValueError("Budget Exceeded")
                        return eval_func(x)
                    
                    try:
                        res = minimize(local_func, best_x.copy(), method='Nelder-Mead',
                                       options={'maxiter': max(1, budget_polish // 4), 
                                                'xatol': 1e-7, 'fatol': 1e-7})
                    except Exception:
                        pass

        return best_x, best_f
