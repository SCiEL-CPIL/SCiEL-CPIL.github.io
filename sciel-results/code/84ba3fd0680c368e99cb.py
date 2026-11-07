# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.X = []
        self.Y = []
        self.rng = np.random.default_rng()
        self.best_f = np.inf
        self.best_x = None

    def __call__(self, func):
        # Phase 1: Initial random sampling
        n_init = 20
        for _ in range(n_init):
            x = self.rng.uniform(-5, 5, self.dim)
            f = func(x)
            self.calls += 1
            self.X.append(x)
            self.Y.append(f)
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
        
        if self.calls >= self.budget:
            return self.best_x, self.best_f

        # GP Parameters
        # We will tune length_scale l and noise sigma_n
        # Normalize inputs to [-1, 1] for kernel stability
        def normalize(x):
            return (x + 5) / 10 * 2 - 1 # Maps [-5,5] to [-1,1]
        
        def denormalize(z):
            return (z + 1) / 2 * 10 - 5 # Maps [-1,1] to [-5,5]

        # Store normalized X for GP
        X_norm = [normalize(x) for x in self.X]
        Y = np.array(self.Y)
        
        # Standardize Y for GP fitting
        Y_mean = np.mean(Y)
        Y_std = np.std(Y)
        if Y_std < 1e-10: Y_std = 1.0
        Y_norm = (Y - Y_mean) / Y_std

        # Hyperparameter search space
        l_options = [0.2, 0.5, 1.0, 2.0]
        sn_options = [1e-4, 1e-2]
        
        best_l = 1.0
        best_sn = 1e-4
        best_log_marginal = -np.inf

        # Quick hyperparameter tuning
        for l in l_options:
            for sn in sn_options:
                try:
                    # Compute Kernel Matrix
                    # X_norm is list of arrays
                    Xn = np.array(X_norm)
                    # Squared distances
                    sq_dists = np.sum((Xn[:, np.newaxis, :] - Xn[np.newaxis, :, :])**2, axis=2)
                    K = np.exp(-sq_dists / (2 * l**2))
                    K[np.diag_indices_from(K)] += sn
                    Yn = Y_norm
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Yn))
                    log_det = 2 * np.sum(np.log(np.diag(L)))
                    log_marginal = -0.5 * Yn @ alpha - 0.5 * log_det - 0.5 * len(Yn) * np.log(2 * np.pi)
                    
                    if log_marginal > best_log_marginal:
                        best_log_marginal = log_marginal
                        best_l = l
                        best_sn = sn
                except:
                    continue

        l = best_l
        sn = best_sn

        # Precompute Cholesky for current best hyperparams
        Xn = np.array(X_norm)
        sq_dists = np.sum((Xn[:, np.newaxis, :] - Xn[np.newaxis, :, :])**2, axis=2)
        K = np.exp(-sq_dists / (2 * l**2))
        K[np.diag_indices_from(K)] += sn
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))

        def predict_gp(z):
            """
            z: array of shape (dim,) or (n, dim)
            Returns mean, var
            """
            if z.ndim == 1:
                z = z.reshape(1, -1)
            
            # Kernel between z and Xn
            sq_dists = np.sum((z[:, np.newaxis, :] - Xn[np.newaxis, :, :])**2, axis=2)
            k = np.exp(-sq_dists / (2 * l**2))
            
            # Mean
            mu_norm = k @ alpha
            mu = mu_norm * Y_std + Y_mean
            
            # Variance
            k_ss = np.exp(0) # k(z,z) = 1
            var_norm = k_ss - np.sum((L.T @ np.linalg.solve(L, k.T))**2, axis=0)
            # Ensure non-negative
            var_norm = np.maximum(var_norm, 0)
            var = var_norm * Y_std**2
            
            if z.shape[0] == 1:
                return mu[0], var[0]
            return mu, var

        def expected_improvement(z):
            """
            z: array of shape (dim,)
            Returns EI
            """
            mu, var = predict_gp(z)
            sigma = np.sqrt(var)
            if sigma < 1e-10:
                return 0.0
            
            # We are minimizing, so EI is based on improvement over best_f
            # Standard EI for minimization:
            # Z = (best_f - mu) / sigma
            Z = (self.best_f - mu) / sigma
            ei = (self.best_f - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
            return ei

        def neg_ei(z):
            return -expected_improvement(z)

        # Loop for remaining budget
        while self.calls < self.budget:
            # 1. Coarse grid search for EI max
            # 5 points per dim -> 3125 points
            grid_points = []
            # Use a simple product of linspace
            coords = np.linspace(-1, 1, 5) # In normalized space
            # Generate grid
            grid = np.array(np.meshgrid(*([coords]*self.dim))).T.reshape(-1, self.dim)
            
            # Evaluate EI on grid
            ei_vals = np.array([neg_ei(z) for z in grid])
            
            # Get top 5 candidates
            top_indices = np.argsort(ei_vals)[:5]
            candidates = grid[top_indices]
            
            # 2. Refine candidates with L-BFGS-B
            best_candidate = None
            best_neg_ei = np.inf
            
            for c in candidates:
                try:
                    res = minimize(neg_ei, c, method='L-BFGS-B', bounds=[(-1, 1)]*self.dim)
                    if res.fun < best_neg_ei:
                        best_neg_ei = res.fun
                        best_candidate = res.x
                except:
                    pass
            
            if best_candidate is None:
                # Fallback to best grid point
                best_candidate = grid[np.argmin(ei_vals)]
            
            # Convert back to original space
            x_next = denormalize(best_candidate)
            
            # Evaluate
            f_next = func(x_next)
            self.calls += 1
            self.X.append(x_next)
            self.Y.append(f_next)
            self.X_norm = [normalize(x) for x in self.X] # Update list
            X_norm = [normalize(x) for x in self.X] # Local var update for next iteration
            
            if f_next < self.best_f:
                self.best_f = f_next
                self.best_x = x_next.copy()
                
            # Update Y stats
            Y = np.array(self.Y)
            Y_mean = np.mean(Y)
            Y_std = np.std(Y)
            if Y_std < 1e-10: Y_std = 1.0
            Y_norm = (Y - Y_mean) / Y_std
            
            # Re-fit GP with new data (optional: only every few steps to save time, but N is small)
            # We re-tune hyperparams every 10 steps to adapt
            if self.calls % 10 == 0 or self.calls == self.budget:
                 # Re-run the quick hyperparameter tuning
                 best_l = 1.0
                 best_sn = 1e-4
                 best_log_marginal = -np.inf
                 for l in l_options:
                     for sn in sn_options:
                         try:
                             Xn_temp = np.array(X_norm)
                             sq_dists_temp = np.sum((Xn_temp[:, np.newaxis, :] - Xn_temp[np.newaxis, :, :])**2, axis=2)
                             K_temp = np.exp(-sq_dists_temp / (2 * l**2))
                             K_temp[np.diag_indices_from(K_temp)] += sn
                             L_temp = np.linalg.cholesky(K_temp)
                             alpha_temp = np.linalg.solve(L_temp.T, np.linalg.solve(L_temp, Y_norm))
                             log_det_temp = 2 * np.sum(np.log(np.diag(L_temp)))
                             log_marginal_temp = -0.5 * Y_norm @ alpha_temp - 0.5 * log_det_temp - 0.5 * len(Y_norm) * np.log(2 * np.pi)
                             
                             if log_marginal_temp > best_log_marginal:
                                 best_log_marginal = log_marginal_temp
                                 best_l = l
                                 best_sn = sn
                         except:
                             continue
                 l = best_l
                 sn = best_sn

            # Recompute Cholesky and alpha for next iteration
            Xn = np.array(X_norm)
            sq_dists = np.sum((Xn[:, np.newaxis, :] - Xn[np.newaxis, :, :])**2, axis=2)
            K = np.exp(-sq_dists / (2 * l**2))
            K[np.diag_indices_from(K)] += sn
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
            except:
                # If Cholesky fails, add more noise
                K[np.diag_indices_from(K)] += 1e-2
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
