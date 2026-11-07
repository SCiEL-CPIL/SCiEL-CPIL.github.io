import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """
    Hierarchical Bayesian Optimization for BBOB f21 (Gallagher 101 peaks).
    
    Strategy:
    1. Initial space-filling sampling to seed the GP.
    2. Iterative BO loop: fit Matern 5/2 GP, maximize Expected Improvement
       (or a mix of EI and UCB) via a local surrogate optimizer, evaluate.
    3. Adaptive length-scale: start broad to find basins, then narrow to refine.
    4. Fallback to (1+1)-ES if GP fitting becomes too slow or budget runs low.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            if not check_time():
                return None
            try:
                val = float(func(x))
            except Exception:
                return None
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)

        # Matern 5/2 kernel
        def matern52_kernel(X1, X2, length_scale, variance):
            """Compute Matern 5/2 kernel matrix between X1 (n1 x d) and X2 (n2 x d)."""
            # Squared distances
            d = np.sqrt(np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] 
                         - 2 * X1 @ X2.T)
            r = d / length_scale
            # Matern 5/2: (1 + sqrt(5)*r + 5*r^2/3) * exp(-sqrt(5)*r)
            kernel = (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r)
            return variance * kernel

        def fit_gp(X, y, length_scale, variance, noise):
            """Fit GP and return kernel matrix, inverse, and alpha."""
            n = len(y)
            K = matern52_kernel(X, X, length_scale, variance)
            K[np.diag_indices(n)] += noise
            try:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
                K_inv = np.linalg.inv(L @ L.T)
                alpha = np.linalg.solve(L, y)
                alpha = np.linalg.solve(L.T, alpha)
                return K, L, K_inv, alpha
            except np.linalg.LinAlgError:
                # Fallback: add jitter
                K += 1e-6 * np.eye(n)
                L = np.linalg.cholesky(K)
                K_inv = np.linalg.inv(L @ L.T)
                alpha = np.linalg.solve(L, y)
                alpha = np.linalg.solve(L.T, alpha)
                return K, L, K_inv, alpha

        def gp_predict(X_new, X_train, y_train, L, alpha, length_scale, variance):
            """GP mean and std prediction."""
            K_new = matern52_kernel(X_new, X_train, length_scale, variance)
            mu = K_new @ alpha
            # Variance: k(x,x) - k(x,X) K^-1 k(X,x)
            v = np.linalg.solve(L, K_new.T)
            var = variance - np.sum(K_new * v.T, axis=1)
            var = np.maximum(var, 1e-12)
            std = np.sqrt(var)
            return mu, std

        def expected_improvement(mu, std, best_f_val, xi=0.01):
            """Compute EI for minimization."""
            improvement = best_f_val - mu - xi
            with np.errstate(divide='ignore', invalid='ignore'):
                z = improvement / (std + 1e-12)
                ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei = np.where(std > 1e-8, ei, 0.0)
            return ei

        def ucb(mu, std, best_f_val, kappa=1.0):
            """Upper Confidence Bound (for minimization, we want lower bound)."""
            return mu - kappa * std

        # Phase 1: Initial space-filling sampling
        n_initial = min(30, budget // 3)
        X_samples = []
        y_samples = []
        
        for i in range(n_initial):
            if not check_time() or calls_made >= budget:
                break
            # Use a quasi-random / stratified approach
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is None:
                break
            X_samples.append(x.copy())
            y_samples.append(fx)

        if best_x is None:
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is not None:
                best_f = fx
                best_x = x.copy()
                X_samples = [x.copy()]
                y_samples = [fx]

        # Phase 2: Iterative Bayesian Optimization
        # Adaptive parameters
        length_scale = 2.5  # Start broad
        variance = 1.0
        noise = 1e-6
        kappa = 2.0  # Exploration-exploitation balance
        
        # Number of candidate points for surrogate optimization
        n_candidates = 200
        n_local_iter = 20  # Local optimization iterations per candidate
        
        bo_iter = 0
        max_bo_iters = budget - calls_made - 50  # Reserve some for final refinement
        
        while check_time() and calls_made < budget - 20 and bo_iter < max_bo_iters:
            bo_iter += 1
            n_train = len(y_samples)
            
            if n_train < 5:
                # Not enough data for GP yet, do more random sampling
                x = rng.uniform(-5, 5, dim)
                fx = safe_call(x)
                if fx is None:
                    break
                X_samples.append(x.copy())
                y_samples.append(fx)
                continue
            
            # Fit GP
            X_train = np.array(X_samples)
            y_train = np.array(y_samples)
            
            # Standardize y for better GP conditioning
            y_mean = np.mean(y_train)
            y_std = np.std(y_train) + 1e-12
            y_norm = (y_train - y_mean) / y_std
            
            K, L, K_inv, alpha = fit_gp(X_train, y_norm, length_scale, 1.0, noise)
            
            # Generate candidate points: mix of random and local perturbations around best
            candidates = []
            
            # Random candidates
            n_rand = n_candidates // 2
            for _ in range(n_rand):
                x = rng.uniform(-5, 5, dim)
                candidates.append(x)
            
            # Local candidates around best points
            n_local = n_candidates - n_rand
            best_indices = np.argsort(y_train)[:min(5, n_train)]
            for idx in best_indices:
                x_base = X_train[idx]
                n_per_point = max(1, n_local // len(best_indices))
                for _ in range(n_per_point):
                    x = x_base + rng.normal(0, length_scale * 0.5, dim)
                    x = np.clip(x, -5, 5)
                    candidates.append(x)
            
            candidates = np.array(candidates)
            
            # Predict on candidates
            mu, std = gp_predict(candidates, X_train, y_norm, L, alpha, length_scale, 1.0)
            
            # Compute acquisition: mix of EI and UCB
            best_f_norm = np.min(y_train)
            ei = expected_improvement(mu, std, best_f_norm)
            ucb_val = ucb(mu, std, best_f_norm, kappa)
            
            # Blend: early on use more UCB (exploration), later use more EI (exploitation)
            progress = min(1.0, bo_iter / max(1, max_bo_iters * 0.5))
            alpha_blend = 0.3 + 0.7 * progress  # 0.3 to 1.0 weight on EI
            acquisition = alpha_blend * ei + (1 - alpha_blend) * (-ucb_val)
            
            # Select best candidate
            best_idx = np.argmax(acquisition)
            x_new = candidates[best_idx]
            
            # Local refinement of the selected candidate using the GP surrogate
            x_opt = x_new.copy()
            for _ in range(n_local_iter):
                if not check_time():
                    break
                # Generate local perturbations
                n_local_cand = 20
                local_cands = x_opt + rng.normal(0, length_scale * 0.3, (n_local_cand, dim))
                local_cands = np.clip(local_cands, -5, 5)
                
                mu_loc, std_loc = gp_predict(local_cands, X_train, y_norm, L, alpha, length_scale, 1.0)
                ei_loc = expected_improvement(mu_loc, std_loc, best_f_norm)
                ucb_loc = ucb(mu_loc, std_loc, best_f_norm, kappa)
                acq_loc = alpha_blend * ei_loc + (1 - alpha_blend) * (-ucb_loc)
                
                best_local_idx = np.argmax(acq_loc)
                x_candidate = local_cands[best_local_idx]
                
                # Check if this is better
                mu_cand, std_cand = gp_predict(x_candidate.reshape(1, -1), X_train, y_norm, L, alpha, length_scale, 1.0)
                mu_best, std_best = gp_predict(x_opt.reshape(1, -1), X_train, y_norm, L, alpha, length_scale, 1.0)
                
                if mu_cand[0] < mu_best[0] - 0.01 * y_std:
                    x_opt = x_candidate
                else:
                    break
            
            # Evaluate the optimized point
            fx_new = safe_call(x_opt)
            if fx_new is None:
                break
            
            X_samples.append(x_opt.copy())
            y_samples.append(fx_new)
            
            # Adapt length scale: shrink as we make progress
            if fx_new < best_f:
                length_scale *= 0.95  # Shrink to refine
            else:
                length_scale *= 1.02  # Slightly expand if no improvement
            
            length_scale = np.clip(length_scale, 0.3, 3.0)
            
            # Adapt kappa: reduce exploration over time
            kappa = max(0.5, 2.0 * (1.0 - bo_iter / max(1, max_bo_iters)))
        
        # Phase 3: Final refinement with (1+1)-ES around the best point
        if best_x is not None and check_time() and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            
            sigma = 0.2
            no_improve = 0
            max_no_improve = 80
            
            while check_time() and calls_made < budget:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma *= 1.03
                else:
                    sigma *= 0.97
                    no_improve += 1
                
                sigma = max(sigma, 1e-6)
                sigma = min(sigma, 2.0)
                
                if no_improve > max_no_improve:
                    # Try a larger step to escape local optimum
                    sigma = 0.8
                    no_improve = 0

        return best_x, best_f
