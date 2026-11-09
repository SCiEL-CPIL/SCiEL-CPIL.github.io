import numpy as np
import time
from scipy.stats import norm
from scipy.spatial.distance import cdist

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape((dim,))
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initialization
        # Use a mix of random points and a small local search to establish initial landscape
        # Since BO is sample efficient, we need a good start.
        # 50 random points + 20 local Nelder-Mead steps? 
        # Let's stick to random for diversity first, maybe 20 points.
        
        # Initialize X and Y
        X = []
        Y = []
        
        # Generate initial points
        # 1. Origin
        # 2. Random uniform
        # 3. Random Gaussian (clipped)
        
        init_points = []
        init_points.append(np.zeros(dim))
        
        # Add 19 random uniform points
        for _ in range(19):
            x = np.random.uniform(-5, 5, dim)
            init_points.append(x)
        
        # Evaluate initial points
        for x in init_points:
            if time.time() > deadline or calls_made >= budget:
                break
            f = evaluate(x)
            X.append(x)
            Y.append(f)
            
        X = np.array(X)
        Y = np.array(Y)
        
        # If we didn't get enough points, stop
        if len(X) < 2:
            return best_x, best_f

        # Phase 2: Bayesian Optimization Loop
        # Hyperparameters for GP
        # We use a Matern 5/2 kernel with ARD (Automatic Relevance Determination)
        # To keep it fast and robust without full MCMC, we use simple grid search or fixed reasonable params.
        # Given 500 budget, we can afford some hyperparameter tuning if it's fast.
        
        # For simplicity and speed, we'll use fixed hyperparameters tuned for 5D box [-5,5].
        # Length scales: start at 2.0 (covers half the domain)
        # Signal variance: 1.0
        # Noise variance: 1e-8 (assume deterministic)
        
        length_scale = 2.0
        sigma_f = 1.0
        sigma_n = 1e-8
        
        # Acquisition: Expected Improvement (EI)
        # xi = 0.01 for exploration
        
        def matern52_kernel(X1, X2, ls, sf):
            # X1: (n1, d), X2: (n2, d)
            # Scale inputs
            X1_s = X1 / ls
            X2_s = X2 / ls
            dists = cdist(X1_s, X2_s)
            sqrt3_dists = np.sqrt(3) * dists
            K = sf**2 * (1 + sqrt3_dists) * np.exp(-sqrt3_dists)
            return K

        def compute_ei(X, Y, X_cand, ls, sf, sn, xi=0.01):
            # X: (n, d), Y: (n,)
            # X_cand: (m, d)
            n = len(X)
            m = len(X_cand)
            
            K = matern52_kernel(X, X, ls, sf)
            K[np.diag_indices_from(K)] += sn
            Ks = matern52_kernel(X_cand, X, ls, sf)
            
            # Solve K theta = Y
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
            except np.linalg.LinAlgError:
                # Fallback to pinv if Cholesky fails
                alpha = np.linalg.pinv(K) @ Y
                L = np.linalg.cholesky(K + 1e-6 * np.eye(n)) # For sigma calc fallback? No, just return 0 EI if unstable?
                # Let's just use pinv for alpha and ignore covariance uncertainty if it fails
                Ks_inv = np.linalg.pinv(K)
                Kss = matern52_kernel(X_cand, X_cand, ls, sf)
                cov = Kss - Ks @ Ks_inv @ Ks.T
                var = np.clip(np.diag(cov), 1e-12, None)
                mean = Ks @ alpha
            else:
                Kss = matern52_kernel(X_cand, X_cand, ls, sf)
                cov = Kss - Ks @ np.linalg.solve(L.T, np.linalg.solve(L, Ks.T))
                var = np.clip(np.diag(cov), 1e-12, None)
                mean = Ks @ alpha
                
            y_min = np.min(Y)
            imp = y_min - xi - mean
            sigma = np.sqrt(var)
            
            # EI = imp * Phi(imp/sigma) + sigma * phi(imp/sigma)
            z = imp / sigma
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            
            return ei

        # Main Loop
        while calls_made < budget and time.time() < deadline:
            # 1. Generate candidates
            # Use a mix of random and local perturbations around best points to ensure coverage
            # For BO, random sampling of acquisition function is standard.
            # We sample 2000 candidates from Uniform[-5,5] and pick the best EI.
            
            n_candidates = 2000
            X_cand = np.random.uniform(-5, 5, (n_candidates, dim))
            
            # Calculate EI for all candidates
            ei_vals = compute_ei(X, Y, X_cand, length_scale, sigma_f, sigma_n)
            
            # Select best candidate
            best_idx = np.argmax(ei_vals)
            x_next = X_cand[best_idx]
            
            # Evaluate
            f_next = evaluate(x_next)
            
            # Update X, Y
            X = np.vstack([X, x_next])
            Y = np.append(Y, f_next)
            
            # Optional: Adaptive Hyperparameters
            # If we have enough points, we could refine length_scale.
            # But fixed LS=2.0 is usually okay for 5D [-5,5].
            # If stagnation occurs, we might increase LS (smoother) or decrease (rougher).
            # Let's stick to fixed for stability in this short budget.
            
            # Convergence check
            if np.std(Y[-10:]) < 1e-6:
                break

        return best_x, best_f
