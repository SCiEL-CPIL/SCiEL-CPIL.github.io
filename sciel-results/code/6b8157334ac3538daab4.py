"""Surrogate-assisted Bayesian optimization for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.gaussian_process import GaussianProcessRegressor
from scipy.gaussian_process.kernels import RBF


class Optimizer:
    """Bayesian optimization with GP surrogate and Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _get_gp_model(self, X, y, length_scale):
        """
        Create and fit a GP model on a subset of data.
        """
        n = X.shape[0]
        # Subsampling strategy: keep top 50 best and random 100 others
        max_train = 150
        if n > max_train:
            y_sorted_idx = np.argsort(y)
            n_keep_best = min(50, n)
            best_indices = y_sorted_idx[:n_keep_best]
            remaining_indices = np.setdiff1d(np.arange(n), best_indices)
            n_random = max_train - n_keep_best
            if n_random > 0 and len(remaining_indices) > 0:
                rand_indices = remaining_indices[np.random.choice(len(remaining_indices), n_random, replace=False)]
            else:
                rand_indices = np.array([], dtype=int)
            selected = np.concatenate([best_indices, rand_indices])
            X_sub = X[selected]
            y_sub = y[selected]
        else:
            X_sub = X
            y_sub = y

        # Standardize y for GP stability
        y_mean = np.mean(y_sub)
        y_std = np.std(y_sub) + 1e-10
        y_norm = (y_sub - y_mean) / y_std

        kernel = RBF(length_scale=length_scale)
        gp = GaussianProcessRegressor(kernel=kernel, noise=1e-6, normalize_y=True)
        
        try:
            gp.fit(X_sub, y_norm)
            return gp, y_mean, y_std
        except Exception:
            return None, y_mean, y_std

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Use global numpy random for diversity across instances
        # Note: np.random is seeded by the environment or system entropy
        n_init = min(30, budget)
        
        # Latin Hypercube Sampling for initialization
        X = np.zeros((n_init, dim))
        for d in range(dim):
            perm = np.random.permutation(n_init)
            X[:, d] = (perm + 0.5) / n_init * 10.0 - 5.0

        y = np.zeros(n_init)
        for i in range(n_init):
            y[i] = func(X[i])

        calls_made = n_init
        best_idx = np.argmin(y)
        best_x = X[best_idx].copy()
        best_y = y[best_idx]
        
        # Adaptive length scale
        # Estimate initial length scale from data spread
        # If points are close, length scale should be small
        if n_init > 1:
            dists = []
            for i in range(n_init):
                for j in range(i+1, n_init):
                    dists.append(np.linalg.norm(X[i] - X[j]))
            length_scale = max(0.5, np.median(dists) * 0.5)
        else:
            length_scale = 1.0

        while calls_made < budget:
            # Generate candidates
            n_candidates = 200
            
            # 1. Random global
            n_global = int(n_candidates * 0.3)
            X_global = np.random.uniform(-5, 5, (n_global, dim))
            
            # 2. Local around best
            n_local = int(n_candidates * 0.4)
            # Adaptive radius: shrink as we get better? 
            # Simple heuristic: radius proportional to current best value magnitude or just fixed small
            radius = max(0.2, length_scale * 0.5)
            X_local = best_x + np.random.normal(0, radius, (n_local, dim))
            X_local = np.clip(X_local, -5, 5)
            
            # 3. Local around second best (if available)
            n_local2 = n_candidates - n_global - n_local
            if n_local2 > 0 and len(y) > 1:
                # Get second best
                sorted_idx = np.argsort(y)
                second_idx = sorted_idx[1]
                second_x = X[second_idx]
                X_local2 = second_x + np.random.normal(0, radius, (n_local2, dim))
                X_local2 = np.clip(X_local2, -5, 5)
            else:
                X_local2 = np.random.uniform(-5, 5, (n_local2, dim))
            
            X_cand = np.vstack([X_global, X_local, X_local2])
            
            # Fit GP
            gp, y_mean, y_std = self._get_gp_model(X, y, length_scale)
            
            if gp is not None:
                # Predict
                mu, sigma = gp.predict(X_cand, return_std=True)
                # Denormalize prediction
                mu = mu * y_std + y_mean
                sigma = sigma * y_std
                
                # Expected Improvement
                improvement = best_y - mu
                z = improvement / (sigma + 1e-8)
                EI = improvement * norm.cdf(z) + sigma * norm.pdf(z)
                EI = np.where(sigma > 1e-8, EI, 0.0)
                
                # Add small exploration bonus for unexplored areas?
                # Simple distance to nearest observed point
                # This is O(N*M) which is 500*200 = 100k ops, acceptable
                dists = np.linalg.norm(X_cand[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)
                min_dists = np.min(dists, axis=1)
                # Normalize distance bonus
                EI += 0.01 * min_dists / (length_scale + 1e-8)
                
                idx = np.argmax(EI)
            else:
                # Fallback: random selection
                idx = np.random.randint(0, len(X_cand))
            
            x_new = X_cand[idx]
            f_new = func(x_new)
            calls_made += 1
            
            X = np.vstack([X, x_new])
            y = np.append(y, f_new)
            
            if f_new < best_y:
                best_y = f_new
                best_x = x_new.copy()
                
            # Adapt length scale every 20 calls
            if calls_made % 20 == 0 and len(y) > 10:
                sorted_idx = np.argsort(y)
                top_k = min(5, len(y))
                best_points = X[sorted_idx[:top_k]]
                dists = []
                for i in range(top_k):
                    for j in range(i + 1, top_k):
                        dists.append(np.linalg.norm(best_points[i] - best_points[j]))
                if dists:
                    new_ls = max(0.1, np.mean(dists) * 0.5)
                    length_scale = 0.7 * length_scale + 0.3 * new_ls

        return best_x, best_y
# EVOLVE-BLOCK-END
