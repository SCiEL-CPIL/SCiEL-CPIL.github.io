import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc, norm

class Optimizer:
    """
    Optimizer using Gaussian Process surrogate with Expected Improvement.
    1. Initial exploration with Latin Hypercube Sampling.
    2. Fit GP surrogate, use EI acquisition for model-guided search.
    3. Adaptive local polish with Nelder-Mead and final high-precision L-BFGS-B pass.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _gp_predict(self, X_train, y_train, X_pred, length_scale=1.0, noise=1e-6):
        """
        Simple GP with RBF kernel: returns mean and std at X_pred.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        X_pred = np.asarray(X_pred, dtype=np.float64)
        
        n_train = X_train.shape[0]
        
        # RBF kernel: K(x, x') = exp(-||x - x'||^2 / (2 * l^2))
        def rbf_kernel(A, B, l):
            # Compute pairwise squared distances
            A2 = np.sum(A**2, axis=1, keepdims=True)
            B2 = np.sum(B**2, axis=1, keepdims=True)
            D2 = A2 + B2.T - 2 * A @ B.T
            D2 = np.maximum(D2, 0)
            return np.exp(-D2 / (2 * l**2))
        
        K = rbf_kernel(X_train, X_train, length_scale)
        K += noise * np.eye(n_train)
        
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            K += 1e-4 * np.eye(n_train)
            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                # Fallback: return mean of training data, high uncertainty
                return np.mean(y_train) * np.ones(X_pred.shape[0]), np.std(y_train) * np.ones(X_pred.shape[0])
        
        # GP mean and variance
        K_pred = rbf_kernel(X_pred, X_train, length_scale)
        mu = K_pred @ K_inv @ y_train
        v = K_pred @ K_inv @ K_pred.T
        var = 1.0 - np.diag(v)
        var = np.maximum(var, 1e-8)
        std = np.sqrt(var)
        
        return mu, std

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        
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
        n_explore = min(25, total_budget)
        
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
        # Adaptive polish: start with frequent polish, reduce as convergence improves
        polish_count = 0
        last_polish_at = 0
        polish_interval = 20  # initial interval
        min_polish_interval = 8  # minimum interval between polishes
        
        iteration = 0
        while calls < total_budget and time.time() < deadline:
            iteration += 1
            
            # Check if we have enough data to fit surrogate
            if len(X_data) < 4:
                # Not enough data, add random points
                n_add = min(8, total_budget - calls)
                for _ in range(n_add):
                    if calls >= total_budget or time.time() > deadline:
                        break
                    p = np.random.uniform(-5, 5, n)
                    eval_func(p)
                continue
            
            # Fit GP surrogate
            X_arr = np.array(X_data)
            f_arr = np.array(f_data)
            
            # Normalize for better GP conditioning
            f_mean = np.mean(f_arr)
            f_std = np.std(f_arr) if np.std(f_arr) > 1e-10 else 1.0
            f_norm = (f_arr - f_mean) / f_std
            
            # Adaptive length scale: start broad, narrow as we converge
            # Use convergence progress to adjust
            progress = min(1.0, calls / total_budget)
            length_scale = 3.0 * (1.0 - 0.5 * progress) + 0.5  # from 3.5 to 2.0
            
            try:
                mu_pred, std_pred = self._gp_predict(X_arr, f_norm, X_arr, length_scale=length_scale, noise=1e-4)
            except Exception:
                # Can't fit, do random search
                p = np.random.uniform(-5, 5, n)
                eval_func(p)
                continue
            
            f_best_norm = min(f_norm)
            
            # Generate candidate points for acquisition
            n_candidates = 200
            # Mix: 40% random, 30% perturbed best, 30% perturbed good points
            n_random = n_candidates // 3
            n_perturb_best = n_candidates // 3
            n_perturb_good = n_candidates - n_random - n_perturb_best
            
            candidates = np.random.uniform(-5, 5, (n_random, n))
            
            # Perturb best point with varying scales
            perturb_scales = [0.1, 0.3, 0.5, 0.8]
            perturbations = np.random.normal(0, 1, (n_perturb_best, n))
            scales = np.random.choice(perturb_scales, size=n_perturb_best)
            perturbed_best = best_x + perturbations * scales[:, np.newaxis]
            perturbed_best = np.clip(perturbed_best, -5, 5)
            candidates = np.vstack([candidates, perturbed_best])
            
            # Perturb other good points
            n_good = min(5, len(X_arr))
            good_indices = np.argsort(f_norm)[:n_good]
            for idx in good_indices:
                n_perturb = max(1, n_perturb_good // n_good)
                perturbations = np.random.normal(0, 1, (n_perturb, n))
                scales = np.random.choice([0.2, 0.4, 0.6], size=n_perturb)
                perturbed = X_arr[idx] + perturbations * scales[:, np.newaxis]
                perturbed = np.clip(perturbed, -5, 5)
                candidates = np.vstack([candidates, perturbed])
            
            # Truncate to max candidates
            candidates = candidates[:n_candidates]
            
            # Evaluate GP on candidates
            try:
                mu_cand, std_cand = self._gp_predict(X_arr, f_norm, candidates, length_scale=length_scale, noise=1e-4)
            except Exception:
                mu_cand = np.random.normal(f_best_norm, 0.5, len(candidates))
                std_cand = np.ones(len(candidates)) * 0.5
            
            # Compute Expected Improvement for minimization
            # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
            diff = f_best_norm - mu_cand
            sigma = std_cand + 1e-10
            z = diff / sigma
            ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
            
            # Add small randomization to break ties
            noise = np.random.normal(0, 0.002, len(candidates))
            scores = ei + noise
            
            # Select best candidate
            best_idx = np.argmax(scores)
            next_x = candidates[best_idx]
            
            # Evaluate the selected point
            eval_func(next_x)
            
            # Adaptive local polish
            # Increase polish frequency as we get closer to convergence
            current_interval = max(min_polish_interval, polish_interval - int(progress * 10))
            if (iteration - last_polish_at) >= current_interval and calls < total_budget - 20:
                remaining = total_budget - calls
                if remaining > 20:
                    # Allocate more budget to polish as we progress
                    polish_budget = min(30, max(10, int(remaining * 0.15)))
                    
                    def local_func(x):
                        if calls >= total_budget or time.time() > deadline:
                            raise ValueError("Budget Exceeded")
                        return eval_func(x)
                    
                    try:
                        res = minimize(local_func, best_x.copy(), method='Nelder-Mead',
                                       options={'maxiter': max(1, polish_budget // 3), 
                                                'xatol': 1e-8, 'fatol': 1e-8,
                                                'adaptive': True})
                        last_polish_at = iteration
                        polish_count += 1
                    except Exception:
                        pass

        # Final high-precision polish pass
        remaining = total_budget - calls
        if remaining > 5 and time.time() < deadline - 5:
            def final_func(x):
                if calls >= total_budget or time.time() > deadline:
                    raise ValueError("Budget Exceeded")
                return eval_func(x)
            
            try:
                # Try L-BFGS-B for final precision
                res = minimize(final_func, best_x.copy(), method='L-BFGS-B',
                               bounds=[(-5, 5)] * n,
                               options={'maxiter': min(50, remaining // 2),
                                        'ftol': 1e-12, 'gtol': 1e-12,
                                        'maxls': 20})
            except Exception:
                pass
            
            # If L-BFGS-B didn't help, try Nelder-Mead with tighter tolerance
            if calls < total_budget - 3 and time.time() < deadline - 3:
                try:
                    res = minimize(final_func, best_x.copy(), method='Nelder-Mead',
                                   options={'maxiter': min(30, total_budget - calls),
                                            'xatol': 1e-10, 'fatol': 1e-10})
                except Exception:
                    pass

        return best_x, best_f
