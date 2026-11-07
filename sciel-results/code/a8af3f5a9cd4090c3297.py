import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        
        # 1. Sobol Initialization
        n_sobol = min(50, budget)
        sobol = qmc.Sobol(d=dim, scramble=False)
        # Generate Sobol points in [0, 1]
        # We need a power of 2 for optimal properties, but any number works
        sobol_points_unit = sobol.random(n_sobol)
        # Map to [-5, 5]
        sobol_points = sobol_points_unit * 10.0 - 5.0
        
        X = []
        Y = []
        
        # Evaluate Sobol points
        for x in sobol_points:
            if time.time() > deadline:
                break
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            X.append(x_clipped)
            Y.append(f_val)
            
        if not X:
            # Fallback if no calls made (shouldn't happen)
            x0 = np.zeros(dim)
            f0 = func(x0)
            X.append(x0)
            Y.append(f0)

        X = np.array(X)
        Y = np.array(Y)
        
        best_f = np.min(Y)
        best_x = X[np.argmin(Y)]
        
        # GP Parameters
        # Matern 5/2 kernel is good for non-smooth, but Schaffers is smooth. 
        # Let's use RBF (Gaussian) kernel as it's simpler and smooth.
        # K(x, y) = sigma_f^2 * exp(-0.5 * ||x-y||^2 / l^2)
        # We fix hyperparameters to avoid optimization overhead.
        # Box size is 10 in each dim. Distance scale l=1.0 seems reasonable.
        sigma_f = 1.0
        l = 1.0
        noise = 1e-6
        
        def matern52_kernel(X1, X2):
            # X1: (N, D), X2: (M, D)
            # Compute squared distances
            X1_sq = np.sum(X1**2, axis=1)[:, None]
            X2_sq = np.sum(X2**2, axis=1)[None, :]
            dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
            dist_sq = np.maximum(dist_sq, 0)
            dist = np.sqrt(dist_sq)
            # Matern 5/2
            return np.exp(-dist) * (1 + dist + dist**2 / 3)
        
        def rbf_kernel(X1, X2):
            X1_sq = np.sum(X1**2, axis=1)[:, None]
            X2_sq = np.sum(X2**2, axis=1)[None, :]
            dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
            dist_sq = np.maximum(dist_sq, 0)
            return np.exp(-0.5 * dist_sq / (l**2))

        # We will use RBF for simplicity and speed, with l=1.0
        # If the function is ill-conditioned, l might need to be smaller.
        # Let's try l=0.5.
        l = 0.5
        sigma_f = 1.0
        
        def get_kernel(X1, X2):
            return sigma_f**2 * rbf_kernel(X1, X2)

        n_calls = len(X)
        
        # Helper to update GP and find next point
        while n_calls < budget and time.time() < deadline:
            # Select subset of points for GP to keep matrix size manageable
            # Use the best 100 points
            max_gp_points = 100
            if len(Y) > max_gp_points:
                # Get indices of best 100
                idx = np.argsort(Y)[:max_gp_points]
                X_gp = X[idx]
                Y_gp = Y[idx]
            else:
                X_gp = X
                Y_gp = Y
                
            N_gp = len(X_gp)
            
            # Compute Kernel Matrix
            K = get_kernel(X_gp, X_gp) + noise * np.eye(N_gp)
            
            # Cholesky Decomposition
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Add jitter if singular
                K += 1e-6 * np.eye(N_gp)
                try:
                    L = np.linalg.cholesky(K)
                except:
                    # Fallback: Random point
                    x_next = np.clip(np.random.uniform(-5, 5, dim), -5, 5)
                    f_next = func(x_next)
                    X.append(x_next)
                    Y.append(f_next)
                    n_calls += 1
                    if f_next < best_f:
                        best_f = f_next
                        best_x = x_next
                    continue
            
            # Mean and Variance Prediction
            # mu(x) = k(x)^T * K^{-1} * y
            # var(x) = k(x,x) - k(x)^T * K^{-1} * k(x)
            
            # We need to select the next point.
            # Generate a set of candidate points
            # Use a mix of random and local around best
            n_candidates = 100
            candidates = np.random.uniform(-5, 5, (n_candidates, dim))
            # Add local candidates around best_x
            for _ in range(20):
                candidates = np.vstack([candidates, best_x + np.random.normal(0, 0.5, dim)])
            
            # Compute kernel between candidates and GP points
            K_s = get_kernel(candidates, X_gp)
            
            # Solve K L = K_s.T  => L v = K_s.T
            # v = K^{-1} K_s.T
            try:
                # Solve L v = K_s.T
                v = np.linalg.solve(L, K_s.T)
            except np.linalg.LinAlgError:
                # Fallback
                x_next = np.clip(np.random.uniform(-5, 5, dim), -5, 5)
                f_next = func(x_next)
                X.append(x_next)
                Y.append(f_next)
                n_calls += 1
                if f_next < best_f:
                    best_f = f_next
                    best_x = x_next
                continue
                
            # Predicted Mean
            mu_pred = v.T @ Y_gp
            
            # Predicted Variance
            # var = k(x,x) - sum(v^2)
            # k(x,x) is sigma_f^2 for RBF
            var_pred = sigma_f**2 - np.sum(v**2, axis=1)
            var_pred = np.maximum(var_pred, 1e-8)
            std_pred = np.sqrt(var_pred)
            
            # Expected Improvement
            # EI = (f_best - mu) * Phi((f_best - mu)/std) + std * phi((f_best - mu)/std)
            # Note: We are minimizing.
            # f_best is the current minimum (best_f)
            # But GP is fit on the subset. We should use the global best_f for EI.
            
            z = (best_f - mu_pred) / std_pred
            # Phi and PDF of standard normal
            from scipy.stats import norm
            phi = norm.pdf(z)
            Phi = norm.cdf(z)
            
            EI = (best_f - mu_pred) * Phi + std_pred * phi
            
            # Select candidate with max EI
            idx_best = np.argmax(EI)
            x_next = candidates[idx_best]
            x_next = np.clip(x_next, -5.0, 5.0)
            
            # Evaluate
            f_next = func(x_next)
            X.append(x_next)
            Y.append(f_next)
            n_calls += 1
            
            if f_next < best_f:
                best_f = f_next
                best_x = x_next
                
        return best_x, best_f
