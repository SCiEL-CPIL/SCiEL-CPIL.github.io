"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """
    Multi-objective optimizer using ParEGO with Gaussian Process surrogate.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Leave 10 seconds margin before the 300s limit
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Initialization with Latin Hypercube Sampling
        n_init = 10
        X_lhs = np.zeros((n_init, self.dim))
        for i in range(self.dim):
            perm = np.random.permutation(n_init)
            X_lhs[:, i] = (perm + 0.5) / n_init
        X_init = lo + X_lhs * (hi - lo)
        
        X_data = []
        F_data = []
        used = 0
        
        for x in X_init:
            if used >= self.budget or time.time() > deadline:
                break
            try:
                f = problem(x)
                X_data.append(x)
                F_data.append(f)
                used += 1
            except Exception:
                # If the first evaluation fails, the result is invalid.
                # If later, we keep what we have.
                # Breaking is safest to avoid further undefined behavior.
                break
                
        if used == 0:
            return None

        # 2. ParEGO Loop
        # Divide remaining budget into several runs with different weight vectors
        remaining_budget = self.budget - used
        n_runs = 5
        if remaining_budget <= 0:
            return None
            
        evals_per_run = max(1, remaining_budget // n_runs)
        
        # GP Hyperparameters (fixed for speed, tuned for general smooth functions)
        sigma_f = 1.0
        length_scale = 0.5
        noise = 1e-4
        
        for run_idx in range(n_runs):
            if used >= self.budget or time.time() > deadline:
                break
                
            # Sample random weight vector from Dirichlet distribution (uniform on simplex)
            w = np.random.dirichlet(np.ones(self.n_obj))
            
            # Scalarize existing data
            X_arr = np.array(X_data)
            Y = np.array([np.dot(f, w) for f in F_data])
            
            # Standardize Y for GP stability
            mu_y = np.mean(Y)
            std_y = np.std(Y)
            if std_y < 1e-8:
                std_y = 1.0
            Y_std = (Y - mu_y) / std_y
            
            # Fit GP
            # Kernel: RBF
            # K_ij = sigma_f^2 * exp(-||xi - xj||^2 / (2 * l^2)) + noise * delta_ij
            
            def rbf_kernel(X1, X2, l):
                # X1: (N, D), X2: (M, D)
                # Returns (N, M)
                sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1.dot(X2.T)
                sq_dists = np.maximum(sq_dists, 0)
                return sigma_f**2 * np.exp(-sq_dists / (2 * l**2))

            K = rbf_kernel(X_arr, X_arr, length_scale)
            K[np.diag_indices_from(K)] += noise
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_std))
            except np.linalg.LinAlgError:
                # If Cholesky fails, add more noise and retry once
                K[np.diag_indices_from(K)] += 1e-3
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_std))
                except np.linalg.LinAlgError:
                    # Fallback: random search for this run
                    for _ in range(evals_per_run):
                        if used >= self.budget or time.time() > deadline:
                            break
                        x_new = lo + np.random.random(self.dim) * (hi - lo)
                        try:
                            f_new = problem(x_new)
                            X_data.append(x_new)
                            F_data.append(f_new)
                            used += 1
                        except Exception:
                            break
                    continue

            # Acquisition: Expected Improvement (EI)
            y_best = np.min(Y_std)
            
            for _ in range(evals_per_run):
                if used >= self.budget or time.time() > deadline:
                    break
                    
                # Generate candidate points
                n_candidates = 1000
                X_cand = lo + np.random.random((n_candidates, self.dim)) * (hi - lo)
                
                # Predict mean and variance
                K_s = rbf_kernel(X_cand, X_arr, length_scale)
                
                # Mean: K_s @ alpha
                mu_pred = K_s.dot(alpha)
                
                # Variance: sigma_f^2 - K_s @ L_inv^T @ L_inv @ K_s.T
                # Let V = solve(L, K_s.T)
                V = np.linalg.solve(L, K_s.T)
                var_pred = sigma_f**2 - np.sum(V**2, axis=0)
                var_pred = np.maximum(var_pred, 1e-8)
                std_pred = np.sqrt(var_pred)
                
                # EI Calculation
                # Z = (y_best - mu) / std
                Z = (y_best - mu_pred) / std_pred
                EI = (y_best - mu_pred) * norm.cdf(Z) + std_pred * norm.pdf(Z)
                
                # Select best candidate
                idx_best = np.argmax(EI)
                x_new = X_cand[idx_best]
                
                try:
                    f_new = problem(x_new)
                    X_data.append(x_new)
                    F_data.append(f_new)
                    used += 1
                except Exception:
                    break

        return None
# EVOLVE-BLOCK-END
