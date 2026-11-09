"""Surrogate-assisted derivative-free optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """
    GP-surrogate-assisted Bayesian optimization for the Discus function.
    Uses Expected Improvement acquisition with adaptive local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def _count_call(self, func, x):
        """Evaluate func(x) and track call count."""
        self.calls += 1
        return func(x)

    def _gp_predict(self, X, F, X_new, length_scale=1.5, noise=1e-6):
        """
        Gaussian Process prediction with Matern 5/2 kernel.
        Returns (mean, std) for X_new given training data (X, F).
        """
        X = np.asarray(X, dtype=float)
        F = np.asarray(F, dtype=float)
        X_new = np.asarray(X_new, dtype=float)
        
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        
        n = X.shape[0]
        d = X.shape[1]
        
        # Matern 5/2 kernel
        def matern52(K):
            r = np.sqrt(np.maximum(K, 1e-12))
            return (1.0 + np.sqrt(5.0) * r + 5.0 / 3.0 * r**2) * np.exp(-np.sqrt(5.0) * r)
        
        # Compute kernel matrices
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist2 = np.sum((X[i] - X[j]) ** 2)
                K[i, j] = matern52(np.array([dist2]))[0] * (length_scale ** 2)
        
        # Add noise for numerical stability
        K += noise * np.eye(n)
        
        # Kernel between training and new points
        K_new = np.zeros((n, X_new.shape[0]))
        for i in range(n):
            for j in range(X_new.shape[0]):
                dist2 = np.sum((X[i] - X_new[j]) ** 2)
                K_new[i, j] = matern52(np.array([dist2]))[0] * (length_scale ** 2)
        
        # Standard GP prediction
        try:
            K_inv = np.linalg.inv(K)
            K_new_full = K_new.T @ K_inv
            mean = K_new_full @ F
            K_nn = np.zeros((X_new.shape[0], X_new.shape[0]))
            for i in range(X_new.shape[0]):
                for j in range(X_new.shape[0]):
                    dist2 = np.sum((X_new[i] - X_new[j]) ** 2)
                    K_nn[i, j] = matern52(np.array([dist2]))[0] * (length_scale ** 2)
            var = np.diag(K_nn) - np.sum(K_new_full * K_new, axis=0)
            std = np.sqrt(np.maximum(var, 0))
        except np.linalg.LinAlgError:
            # Fallback: use nearest neighbor interpolation
            mean = np.zeros(X_new.shape[0])
            std = np.ones(X_new.shape[0]) * (F.max() - F.min()) * 0.5
            for j in range(X_new.shape[0]):
                dists = np.sum((X - X_new[j]) ** 2, axis=1)
                idx = np.argmin(dists)
                mean[j] = F[idx]
                std[j] = (F.max() - F.min()) * 0.3
        
        return mean, std

    def _ei_acquisition(self, X, F, x_best, f_best, X_candidates, length_scale=1.5, noise=1e-6, xi=0.01):
        """
        Expected Improvement acquisition function.
        Returns EI values for each candidate in X_candidates.
        """
        X = np.asarray(X, dtype=float)
        F = np.asarray(F, dtype=float)
        X_candidates = np.asarray(X_candidates, dtype=float)
        
        if X_candidates.ndim == 1:
            X_candidates = X_candidates.reshape(1, -1)
        
        mean, std = self._gp_predict(X, F, X_candidates, length_scale, noise)
        
        # EI for minimization
        improvement = f_best - mean - xi
        with np.errstate(divide='ignore', invalid='ignore'):
            z = improvement / (std + 1e-12)
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        
        # Handle cases where std is very small
        mask = std < 1e-8
        ei[mask] = improvement[mask]
        
        return ei

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self.calls = 0

        # Phase 1: Initial space-filling sampling
        n_init = min(15, budget)
        rng = np.random.default_rng(42)
        
        # Use Sobol-like sampling for better coverage
        X = np.zeros((n_init, dim))
        for i in range(n_init):
            # Halving sequence for quasi-random coverage
            for d in range(dim):
                X[i, d] = -5 + 10 * ((i + 1) * (2**(d+1) + 1) / (2**(dim+1))) % 1.0
        
        # Add some pure random points
        n_random = max(1, n_init // 3)
        X[:n_random] = rng.uniform(-5, 5, size=(n_random, dim))
        
        F = np.array([self._count_call(func, x) for x in X])
        
        best_idx = np.argmin(F)
        x_best = X[best_idx].copy()
        f_best = F[best_idx]
        
        # Adaptive parameters
        length_scale = 2.0  # Start with broader length scale
        noise = 1e-5
        xi = 0.01  # Exploitation parameter
        
        # Phase 2: GP-guided iterative optimization
        remaining = budget - self.calls
        n_iter = 0
        stagnant = 0  # Count iterations without improvement
        
        while remaining > 0 and n_iter < 480:
            # Generate candidate points for acquisition
            n_candidates = min(200, remaining * 2)
            
            # Mix of local (around best) and global candidates
            candidates = []
            
            # Local candidates around best point
            n_local = min(100, n_candidates // 2)
            local_scale = max(0.1, 1.0 * np.exp(-0.01 * n_iter))
            for _ in range(n_local):
                x_cand = x_best + local_scale * rng.standard_normal(dim)
                x_cand = np.clip(x_cand, -5, 5)
                candidates.append(x_cand)
            
            # Global candidates
            n_global = min(100, n_candidates - n_local)
            for _ in range(n_global):
                x_cand = rng.uniform(-5, 5, dim)
                candidates.append(x_cand)
            
            candidates = np.array(candidates)
            
            # Compute EI acquisition for all candidates
            try:
                ei_values = self._ei_acquisition(X, F, x_best, f_best, candidates, length_scale, noise, xi)
            except Exception:
                # Fallback: random selection
                idx = rng.integers(len(candidates))
                x_cand = candidates[idx]
                f_cand = self._count_call(func, x_cand)
                X = np.vstack([X, x_cand.reshape(1, -1)])
                F = np.append(F, f_cand)
                if f_cand < f_best:
                    f_best = f_cand
                    x_best = x_cand.copy()
                    stagnant = 0
                else:
                    stagnant += 1
                remaining = budget - self.calls
                n_iter += 1
                continue
            
            # Select candidate with highest EI
            best_cand_idx = np.argmax(ei_values)
            x_cand = candidates[best_cand_idx]
            
            # Evaluate the selected candidate
            f_cand = self._count_call(func, x_cand)
            
            # Update model
            X = np.vstack([X, x_cand.reshape(1, -1)])
            F = np.append(F, f_cand)
            
            if f_cand < f_best:
                f_best = f_cand
                x_best = x_cand.copy()
                stagnant = 0
                # Adapt: increase exploitation as we find improvements
                xi = max(0.001, xi * 0.95)
                length_scale = max(0.3, length_scale * 0.98)
            else:
                stagnant += 1
                # Adapt: increase exploration when stagnant
                if stagnant > 5:
                    xi = min(0.05, xi * 1.2)
                    length_scale = min(5.0, length_scale * 1.05)
                    stagnant = 0
            
            # Adaptive step size for local refinement
            remaining = budget - self.calls
            n_iter += 1
            
            # If we're doing well, do a local refinement step
            if remaining >= 3 and stagnant == 0 and n_iter % 5 == 0:
                # Local Nelder-Mead refinement around best point
                try:
                    def local_obj(x):
                        return self._count_call(func, np.clip(x, -5, 5))
                    
                    res = minimize(
                        local_obj,
                        x_best,
                        method='Nelder-Mead',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 10, 'xatol': 1e-3, 'fatol': 1e-6}
                    )
                    x_refined = np.clip(res.x, -5, 5)
                    f_refined = res.fun
                    
                    if f_refined < f_best:
                        f_best = f_refined
                        x_best = x_refined.copy()
                    
                    X = np.vstack([X, x_refined.reshape(1, -1)])
                    F = np.append(F, f_refined)
                except Exception:
                    pass
        
        return x_best, f_best


# EVOLVE-BLOCK-END
