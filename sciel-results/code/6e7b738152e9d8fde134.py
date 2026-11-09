"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """Surrogate-assisted Bayesian Optimization with GP and EI, followed by CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill
        
        rng = np.random.default_rng(42)
        
        # State
        X = []  # List of evaluated points
        Y = []  # List of evaluated values
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        remaining_budget = self.budget
        
        # Helper to evaluate a point and update state
        def evaluate_point(x):
            nonlocal best_x, best_f, calls_made
            if time.time() > deadline or calls_made >= remaining_budget:
                return
            try:
                fx = func(x)
                calls_made += 1
                X.append(x)
                Y.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            except Exception:
                pass

        # Phase 1: Initial Exploration with LHS (Latin Hypercube Sampling)
        # We need a good initial set of points to train the GP.
        # For 5D, 15 points is a reasonable start.
        n_init = 15
        if n_init > remaining_budget:
            n_init = remaining_budget
        
        # Generate LHS points
        lhs_points = np.zeros((n_init, self.dim))
        for i in range(n_init):
            for d in range(self.dim):
                lhs_points[i, d] = (i + rng.random()) / n_init * 10.0 - 5.0
        
        # Shuffle to avoid order bias
        idx = rng.permutation(n_init)
        lhs_points = lhs_points[idx]
        
        for p in lhs_points:
            evaluate_point(p)
            if time.time() > deadline or calls_made >= remaining_budget:
                break

        if best_x is None:
            # Fallback if no calls were made
            p = rng.uniform(-5, 5, self.dim)
            evaluate_point(p)
            if best_x is None:
                return p, np.inf

        # Phase 2: Bayesian Optimization Loop
        # Use GP to predict and EI to select next point
        # We will run this until we have used ~60% of the budget or time runs out
        bo_budget_limit = int(remaining_budget * 0.6)
        
        # GP Parameters
        # We use a simple RBF kernel with automatic relevance determination (ARD)
        # To keep it fast, we will manually implement the GP prediction using NumPy
        # and a simple optimization for hyperparameters if needed, or fixed hyperparams for speed.
        # Given the 300s limit and small N, we can afford a simple hyperparameter search or fixed good params.
        
        # Fixed hyperparameters for speed and stability in 5D
        # Length scales around 1.0, noise 1e-6, signal variance 1.0
        # We might scale signal variance based on initial Y range
        
        def compute_gp_params(X, Y):
            """Compute GP hyperparameters. For simplicity and speed, we use fixed reasonable values 
            or a very quick optimization. Given N is small, fixed is safer for timeout."""
            # Normalize Y for better conditioning
            y_mean = np.mean(Y)
            y_std = np.std(Y) + 1e-8
            Y_norm = (Y - y_mean) / y_std
            
            # Simple fixed hyperparameters
            # Length scales: 1.0 for all dimensions
            # Signal variance: 1.0 (since Y is normalized)
            # Noise: 1e-4
            return 1.0, 1.0, 1e-4, y_mean, y_std

        def gp_predict(X, Y, X_test, length_scale, signal_var, noise):
            """
            Compute mean and variance of GP at X_test given training data X, Y.
            X: (N, D)
            Y: (N,)
            X_test: (M, D)
            """
            N = X.shape[0]
            D = X.shape[1]
            
            # Kernel function: RBF
            # K(X, X)
            # We compute squared distances efficiently
            # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
            X_sq = np.sum(X**2, axis=1)
            K = X_sq[:, None] + X_sq[None, :] - 2 * X @ X.T
            K = np.maximum(K, 0) # Numerical stability
            K = signal_var * np.exp(-K / (2 * length_scale**2))
            
            # Add noise to diagonal
            K[np.diag_indices_from(K)] += noise
            
            # Cholesky decomposition for stability
            try:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(N))
            except np.linalg.LinAlgError:
                # Fallback to solve with jitter
                L = np.linalg.cholesky(K + 1e-4 * np.eye(N))
                
            # Alpha = K^-1 Y
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
            
            # Kernel between training and test
            X_test_sq = np.sum(X_test**2, axis=1)
            K_s = X_test_sq[:, None] + X_sq[None, :] - 2 * X_test @ X.T
            K_s = np.maximum(K_s, 0)
            K_s = signal_var * np.exp(-K_s / (2 * length_scale**2))
            
            # Mean prediction
            mu = K_s @ alpha
            
            # Variance prediction
            # v = L^-1 K_s^T
            v = np.linalg.solve(L, K_s.T)
            var = signal_var - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10) # Ensure positive variance
            
            return mu, var

        # EI Acquisition Function
        def expected_improvement(mu, var, best_f_current, xi=0.01):
            """
            Calculate Expected Improvement.
            mu: (M,)
            var: (M,)
            best_f_current: scalar (minimum found so far)
            """
            sd = np.sqrt(var)
            # Avoid division by zero
            sd[sd < 1e-10] = 1e-10
            imp = best_f_current - mu - xi
            z = imp / sd
            ei = imp * norm.cdf(z) + sd * norm.pdf(z)
            return ei

        # Candidate generation for BO
        # We will generate a set of candidate points using Sobol/LHS and pick the one with highest EI
        
        while calls_made < bo_budget_limit and time.time() < deadline:
            # Check if we have enough points to train GP (min 2, but 5 is better)
            if len(X) < 5:
                # Just sample random point
                x_new = rng.uniform(-5, 5, self.dim)
                evaluate_point(x_new)
                continue
                
            X_arr = np.array(X)
            Y_arr = np.array(Y)
            
            # Normalize Y for GP stability
            y_mean = np.mean(Y_arr)
            y_std = np.std(Y_arr) + 1e-8
            Y_norm = (Y_arr - y_mean) / y_std
            
            # Get GP parameters
            length_scale, signal_var, noise, _, _ = compute_gp_params(X_arr, Y_norm)
            
            # Generate candidates
            n_candidates = 500
            # Use LHS for candidates to ensure spread
            cand_points = np.zeros((n_candidates, self.dim))
            for i in range(n_candidates):
                for d in range(self.dim):
                    cand_points[i, d] = (i + rng.random()) / n_candidates * 10.0 - 5.0
            
            # Predict
            mu, var = gp_predict(X_arr, Y_norm, cand_points, length_scale, signal_var, noise)
            
            # Denormalize mean and variance for EI calculation?
            # Actually, EI is scale invariant if we scale consistently.
            # Let's work in normalized space for EI, then map back.
            # Best f in normalized space
            best_f_norm = (best_f - y_mean) / y_std
            
            # Calculate EI
            ei = expected_improvement(mu, var, best_f_norm)
            
            # Select best candidate
            idx_best = np.argmax(ei)
            x_new = cand_points[idx_best]
            
            evaluate_point(x_new)
            
            # Optional: If improvement is small and variance is small, we might be converging.
            # But let's let the BO loop run its course.

        # Phase 3: Local Refinement with CMA-ES
        # Use the best point found by BO as the seed for a (1+1)-ES or CMA-ES
        if best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # Start with moderate sigma
            
            # Run (1+1)-ES for the remaining budget
            while calls_made < remaining_budget and time.time() < deadline:
                # Generate neighbor
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                
                evaluate_point(y)
                
                if time.time() > deadline or calls_made >= remaining_budget:
                    break
                    
                # Check if y was actually evaluated (it should be, unless deadline hit)
                # We need to compare fx with the new best if y was better
                # The evaluate_point function updates best_x/best_f if y is better.
                # We need to know if the last call improved the result to adjust sigma.
                # Let's track the last evaluated value separately for sigma adaptation.
                
                # Actually, evaluate_point updates global best. 
                # To adapt sigma, we need to know if *this specific step* was an improvement 
                # relative to the *current* x, not just the global best.
                
                # Let's refactor evaluate_point or handle sigma logic here.
                # Since evaluate_point updates best_x/f, we can compare fx before and after.
                # But we lost the 'fx' of the specific y if it wasn't the global best.
                
                # Let's just use a simple rule: if global best improved, increase sigma? 
                # No, that's dangerous. 
                # Better: Keep a local variable for the current step's result.
                
                # Let's redo the loop structure to be more explicit for the ES part.
                pass 

            # The above loop structure is slightly flawed for sigma adaptation because 
            # evaluate_point hides the individual result. Let's implement a dedicated ES loop.
            
            # Reset for dedicated ES loop
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            
            while calls_made < remaining_budget and time.time() < deadline:
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                
                try:
                    fy = func(y)
                    calls_made += 1
                except Exception:
                    break
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                    # Update global best if this is better
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    
                if sigma < 1e-6:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
