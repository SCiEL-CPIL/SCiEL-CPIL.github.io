import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        # State variables
        best_x = None
        best_fx = np.inf
        calls_made = 0
        X = []  # List of evaluated points
        Y = []  # List of evaluated function values

        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal calls_made, best_x, best_fx
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            X.append(x.copy())
            Y.append(fx)
            return fx

        # Phase 1: Initial Random Sampling
        n_init = min(40, budget)
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            safe_func(x)

        # Phase 2: Surrogate-Assisted Search (GP + UCB)
        # We will use this phase until we have used ~80% of the budget
        # or until time is running low.
        
        # GP Hyperparameters (Fixed for speed and stability)
        # Matern 5/2 kernel is good for smooth functions
        l_kernel = 1.5  # Length scale
        sigma_f = 1.0   # Signal variance
        sigma_n = 1e-4  # Noise variance

        def matern52_kernel(X1, X2, l):
            # X1: (N, dim), X2: (M, dim)
            # dist: (N, M)
            dist = cdist(X1, X2)
            sqrt5_dist = np.sqrt(5) * dist / l
            return (sigma_f ** 2) * (1 + sqrt5_dist) * np.exp(-sqrt5_dist)

        def fit_gp(X_data, Y_data):
            """
            Fits a GP to the data. Returns a function that predicts mean and std for new points.
            """
            X_data = np.array(X_data)
            Y_data = np.array(Y_data)
            N = len(Y_data)
            
            # Kernel matrix
            K = matern52_kernel(X_data, X_data, l_kernel)
            K += sigma_n * np.eye(N)
            
            # Cholesky decomposition for stability
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Add jitter if Cholesky fails
                K += 1e-6 * np.eye(N)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    # Fallback to pinv if still failing (shouldn't happen with jitter)
                    L = np.linalg.pinv(K)
                    return None # Signal failure

            # Pre-compute inverse for prediction
            # alpha = K^-1 y
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_data))
            
            def predict(X_new):
                """
                Predicts mean and std for X_new.
                X_new: (M, dim)
                Returns: mean (M,), std (M,)
                """
                X_new = np.array(X_new)
                if X_new.ndim == 1:
                    X_new = X_new.reshape(1, -1)
                    
                K_s = matern52_kernel(X_data, X_new, l_kernel) # (N, M)
                K_ss = matern52_kernel(X_new, X_new, l_kernel) # (M, M)
                
                # Mean: K(X_new, X) K(X, X)^-1 y
                # Note: K_s is (N, M), we need (M, N) for matrix mult with alpha (N,)
                # Actually, mean = K_s.T @ alpha
                mean = K_s.T @ alpha
                
                # Variance: K(X_new, X_new) - K(X_new, X) K(X, X)^-1 K(X, X_new)
                # K_s.T is (M, N)
                # K_s is (N, M)
                # K(X, X)^-1 is implicit in alpha calculation, but we need the matrix for variance
                # Let's compute K_inv = L^-1 L^-T
                # Var = K_ss - K_s.T @ K_inv @ K_s
                # To save time, we can compute this term directly if we stored L
                # But here we are in a closure. Let's just compute the variance term.
                # Actually, for UCB we need std.
                # var = np.diag(K_ss - K_s.T @ (np.linalg.solve(L.T, np.linalg.solve(L, K_s))))
                # This is O(N^2 M).
                
                try:
                    # Solve L v = K_s
                    v = np.linalg.solve(L, K_s)
                    # Solve L.T w = v
                    w = np.linalg.solve(L.T, v)
                    # K_s.T @ w is (M, M)
                    var = np.diag(K_ss) - np.sum(K_s.T * w, axis=0)
                    var = np.maximum(var, 0) # Numerical stability
                    std = np.sqrt(var)
                except np.linalg.LinAlgError:
                    std = np.ones(X_new.shape[0])
                    mean = np.zeros(X_new.shape[0])
                    
                return mean, std

            return predict

        # Phase 2 Loop
        es_budget = int(budget * 0.8)
        
        while calls_made < es_budget:
            if time.time() > deadline:
                break
                
            if len(X) < 5:
                # Not enough data for GP, just sample randomly
                x = rng.uniform(-5, 5, dim)
                safe_func(x)
                continue

            # Fit GP
            predictor = fit_gp(X, Y)
            if predictor is None:
                # GP fitting failed, fallback to random
                x = rng.uniform(-5, 5, dim)
                safe_func(x)
                continue

            # Generate Candidate Points
            # Mix of local exploration around best and global random
            n_candidates = 200
            candidates = []
            
            # 50% Local around best_x
            n_local = n_candidates // 2
            # Adaptive sigma for local search
            # Start wide, narrow down
            progress = calls_made / es_budget
            local_sigma = 2.0 * (1.0 - progress) + 0.5
            local_cands = best_x + rng.standard_normal((n_local, dim)) * local_sigma
            local_cands = np.clip(local_cands, -5, 5)
            candidates.append(local_cands)
            
            # 30% Random in box
            n_rand = n_candidates // 3
            rand_cands = rng.uniform(-5, 5, (n_rand, dim))
            candidates.append(rand_cands)
            
            # 20% Around current best from previous iterations (if different)
            # Just add more random for simplicity
            
            candidates = np.vstack(candidates)
            
            # Predict Mean and Std
            mean, std = predictor(candidates)
            
            # Acquisition Function: UCB (Upper Confidence Bound)
            # beta = 2 * log(1 + t * pi^2 * D / (6 * log(1 + t / gamma)))
            # Simplified beta for 5D:
            t = len(X)
            D = dim
            gamma = 2.0
            beta = 2 * np.log(1 + t * np.pi**2 * D / (6 * np.log(1 + t / gamma)))
            
            ucb = mean + np.sqrt(beta) * std
            
            # Select best candidate
            best_idx = np.argmax(ucb)
            x_next = candidates[best_idx]
            
            # Evaluate
            safe_func(x_next)

        # Phase 3: Local Refinement
        remaining_budget = budget - calls_made
        if remaining_budget > 5 and best_x is not None and time.time() < deadline:
            # Use Powell or Nelder-Mead
            # We wrap the function to ensure we don't exceed the total budget
            local_calls = 0
            def local_func(x):
                nonlocal local_calls, calls_made
                if local_calls >= remaining_budget:
                    return best_fx
                local_calls += 1
                return safe_func(x)
            
            try:
                # Powell is robust for smooth functions
                res = minimize(
                    local_func, 
                    best_x, 
                    method='Powell', 
                    options={'maxiter': remaining_budget, 'xtol': 1e-12, 'ftol': 1e-12}
                )
                # Update best if improvement found
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_fx
