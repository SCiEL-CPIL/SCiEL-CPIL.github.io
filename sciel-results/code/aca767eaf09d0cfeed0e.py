"""Model-and-optimize optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """GP-based surrogate optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _fit_gp(self, X, Y, sigma_f, sigma_n, length_scale):
        """Fit GP and return kernel matrix and mean/cov parameters."""
        n = len(X)
        # Compute kernel matrix
        # X is (n, d)
        # Use broadcasting
        # D2 = ||x_i - x_j||^2
        sq_norms = np.sum(X ** 2, axis=1)
        D2 = sq_norms[:, None] + sq_norms[None, :] - 2 * X @ X.T
        np.maximum(D2, 0, out=D2)
        
        K = sigma_f ** 2 * np.exp(-0.5 * D2 / (length_scale ** 2))
        K[np.diag_indices_from(K)] += sigma_n
        
        # Cholesky
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Add jitter
            K += 1e-6 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except:
                # Fallback: return None
                return None

        # Solve for alpha
        # alpha = L^-T L^-1 Y
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
        
        return K, L, alpha

    def _gp_predict(self, X_new, X_train, Y_train, L, alpha, sigma_f, sigma_n, length_scale):
        """Predict mean and variance at X_new."""
        # Compute K(X_new, X_train)
        sq_norms_train = np.sum(X_train ** 2, axis=1)
        sq_norms_new = np.sum(X_new ** 2, axis=1)
        D2 = sq_norms_new[:, None] + sq_norms_train[None, :] - 2 * X_new @ X_train.T
        np.maximum(D2, 0, out=D2)
        
        K_new = sigma_f ** 2 * np.exp(-0.5 * D2 / (length_scale ** 2))
        
        # Mean: K_new @ alpha
        mean = K_new @ alpha
        
        # Variance: sigma_f^2 - K_new @ L^-1 L^-T K_new^T
        # Let v = L^-1 K_new^T
        v = np.linalg.solve(L, K_new.T)
        var = sigma_f ** 2 - np.sum(v * K_new.T, axis=1)
        var = np.maximum(var, 1e-10)
        
        return mean, var

    def _expected_improvement(self, x, mean, var, best_f, xi=0.01):
        """Compute EI for a single point x (scalar)."""
        sigma = np.sqrt(var)
        if sigma < 1e-8:
            return 0.0
        z = (best_f - mean - xi) / sigma
        ei = (best_f - mean - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def _gp_search(self, func, budget, rng, deadline):
        dim = self.dim
        n_init = 20
        X = rng.uniform(-5, 5, (n_init, dim))
        Y = np.array([func(x) for x in X])
        evals = n_init
        
        best_idx = np.argmin(Y)
        best_x = X[best_idx].copy()
        best_f = Y[best_idx]
        
        # Hyperparameters
        # Estimate sigma_f
        sigma_f = max(np.std(Y), 1.0)
        sigma_n = 1e-6
        # Estimate length scale
        # Use median distance
        from scipy.spatial.distance import cdist
        dists = cdist(X, X)
        dists = dists[dists > 0]
        length_scale = max(np.median(dists) / 2, 0.5)
        
        while evals < budget:
            if time.time() > deadline:
                break
            
            # Fit GP
            gp_params = self._fit_gp(X, Y, sigma_f, sigma_n, length_scale)
            if gp_params is None:
                break
            K, L, alpha = gp_params
            
            # Generate candidates
            n_candidates = 200
            X_cand = rng.uniform(-5, 5, (n_candidates, dim))
            
            # Predict
            mean, var = self._gp_predict(X_cand, X, Y, L, alpha, sigma_f, sigma_n, length_scale)
            
            # Compute EI
            ei_vals = np.array([self._expected_improvement(x, m, v, best_f) for x, m, v in zip(X_cand, mean, var)])
            
            # Select best candidate
            best_cand_idx = np.argmax(ei_vals)
            x_new = X_cand[best_cand_idx]
            
            # Evaluate
            f_new = func(x_new)
            evals += 1
            
            # Update data
            X = np.vstack([X, x_new])
            Y = np.append(Y, f_new)
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
                
            # Update hyperparameters occasionally?
            # Keep it simple for now.
            
        return best_x, best_f, evals

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x, best_f, evals = self._gp_search(func, budget, rng, deadline)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
