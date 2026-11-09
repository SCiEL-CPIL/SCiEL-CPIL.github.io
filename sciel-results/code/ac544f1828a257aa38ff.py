import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320

        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0

        def check_deadline():
            return time.time() < deadline - 2.0

        # --- Phase 1: Initial Sampling ---
        n_init = min(30, max(10, int(self.budget * 0.1)))
        rng = np.random.default_rng(12345)

        for i in range(n_init):
            if not check_deadline() or eval_count >= self.budget:
                break
            x = rng.uniform(-5.0, 5.0, self.dim)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if eval_count >= self.budget or not check_deadline():
            return best_x, best_f

        # --- Phase 2: Surrogate-Assisted Search ---
        # Use a simple Gaussian Process surrogate to guide the search.
        # This is a structural change from the previous CMA-ES + (1+1)-ES approach.
        # We'll use a simple GP implementation with a squared exponential kernel.
        
        # Store evaluated points
        X = []
        Y = []
        
        # Add the initial samples to the surrogate data
        # We don't have the actual X values from Phase 1, so we need to re-evaluate or store them.
        # To be efficient, let's just start the surrogate phase with the best point found.
        # Actually, we should store the initial points. Let's refactor slightly.
        
        # Re-do Phase 1 with storage
        X_init = []
        Y_init = []
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        
        for i in range(n_init):
            if not check_deadline() or eval_count >= self.budget:
                break
            x = rng.uniform(-5.0, 5.0, self.dim)
            fx = func(x)
            eval_count += 1
            X_init.append(x)
            Y_init.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        X = X_init
        Y = Y_init

        # Simple GP implementation
        def sq_exp_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
            # X1: (n1, d), X2: (n2, d)
            # Compute pairwise squared distances
            X1_sq = np.sum(X1**2, axis=1)[:, None]
            X2_sq = np.sum(X2**2, axis=1)[None, :]
            sq_dists = X1_sq + X2_sq - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return sigma_f**2 * np.exp(-0.5 * sq_dists / length_scale**2)
        
        # Hyperparameters
        length_scale = 0.5
        sigma_f = 1.0
        noise = 1e-6
        
        def predict_gp(X_new, X_train, Y_train):
            # X_new: (m, d), X_train: (n, d), Y_train: (n,)
            K = sq_exp_kernel(X_train, X_train, length_scale, sigma_f)
            K += noise * np.eye(len(X_train))
            Ks = sq_exp_kernel(X_train, X_new, length_scale, sigma_f)
            
            try:
                alpha = np.linalg.solve(K, Y_train)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, Y_train, rcond=None)[0]
            
            mu = Ks.T @ alpha
            Kss = sq_exp_kernel(X_new, X_new, length_scale, sigma_f)
            var = np.diag(Kss) - np.sum(Ks**2 * alpha[None, :], axis=1)
            var = np.maximum(var, 1e-10)
            return mu, np.sqrt(var)
        
        # Surrogate-assisted search loop
        remaining = self.budget - eval_count
        rng_search = np.random.default_rng(54321)
        
        while remaining > 0 and check_deadline():
            # Generate candidate points: mix of exploitation (near best) and exploration
            n_candidates = 10
            
            candidates = []
            # Exploitation: perturb the current best
            for _ in range(n_candidates // 2):
                x = best_x + 0.5 * rng_search.standard_normal(self.dim)
                x = np.clip(x, -5.0, 5.0)
                candidates.append(x)
            
            # Exploration: random points
            for _ in range(n_candidates // 2):
                x = rng_search.uniform(-5.0, 5.0, self.dim)
                candidates.append(x)
            
            candidates = np.array(candidates)
            
            # Predict mean and variance
            if len(X) > 0:
                mu, sigma = predict_gp(candidates, np.array(X), np.array(Y))
                # Expected improvement (EI) for minimization
                # EI = (mu - best_f) * Phi(z) + sigma * phi(z)
                # where z = (mu - best_f) / sigma
                from scipy.stats import norm
                z = (mu - best_f) / (sigma + 1e-10)
                ei = (mu - best_f) * norm.cdf(z) + sigma * norm.pdf(z)
                ei = np.maximum(ei, 0)
                # Select the candidate with highest EI
                idx = np.argmax(ei)
            else:
                idx = 0
            
            x_new = candidates[idx]
            fx_new = func(x_new)
            eval_count += 1
            remaining -= 1
            
            X.append(x_new)
            Y.append(fx_new)
            
            if fx_new < best_f:
                best_f = fx_new
                best_x = x_new.copy()
            
            # Occasionally update hyperparameters based on data
            if len(X) % 20 == 0 and len(X) > 10:
                # Simple adaptation: adjust length_scale based on spread of points
                X_arr = np.array(X)
                mean_dist = np.mean(np.sqrt(np.sum((X_arr[:, None, :] - X_arr[None, :, :])**2, axis=-1)))
                length_scale = max(0.1, min(2.0, mean_dist / 2))
        
        return best_x, best_f
