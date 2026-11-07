import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # Storage for GP
        X = []
        Y = []
        
        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            X.append(pt.copy())
            Y.append(val)
            return val

        def matern52_kernel(X1, X2, length_scale=1.0):
            """
            Compute the Matern 5/2 kernel between two sets of points.
            X1: (n1, d)
            X2: (n2, d)
            Returns: (n1, n2) kernel matrix
            """
            # Compute squared distances
            X1_norm = np.sum(X1**2, axis=1)[:, np.newaxis]
            X2_norm = np.sum(X2**2, axis=1)[np.newaxis, :]
            sq_dists = np.maximum(X1_norm - 2 * X1 @ X2.T + X2_norm, 0)
            dists = np.sqrt(sq_dists)
            
            # Matern 5/2
            sqrt3_dists = np.sqrt(3) * dists / length_scale
            # Avoid division by zero
            sqrt3_dists = np.maximum(sqrt3_dists, 1e-10)
            kernel = (1 + sqrt3_dists + sqrt3_dists**2 / 3) * np.exp(-sqrt3_dists)
            return kernel

        def gp_predict(X_new, length_scale=1.0, noise=1e-10):
            """
            Predict mean and variance at X_new given training data X, Y.
            """
            if len(X) == 0:
                # No data, return prior mean (0) and prior variance (1)
                return np.zeros(X_new.shape[0]), np.ones(X_new.shape[0])
            
            X_train = np.array(X)
            Y_train = np.array(Y)
            
            K = matern52_kernel(X_train, X_train, length_scale)
            K += noise * np.eye(len(X_train))
            K_f = matern52_kernel(X_new, X_train, length_scale)
            
            try:
                # Solve for alpha
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                
                # Predicted mean
                mu = K_f @ alpha
                
                # Predicted variance
                v = np.linalg.solve(L, K_f.T)
                var = 1.0 - np.sum(v**2, axis=0)
                var = np.maximum(var, 0.0)
                
                return mu, var
            except np.linalg.LinAlgError:
                # Fallback to simple interpolation or zero
                return np.zeros(X_new.shape[0]), np.ones(X_new.shape[0])

        def expected_improvement(mu, var, best_y, xi=0.01):
            """
            Compute Expected Improvement.
            """
            var = np.maximum(var, 1e-10)
            sigma = np.sqrt(var)
            z = (best_y - mu - xi) / sigma
            ei = (best_y - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            # For minimization, EI is typically (best - mu - xi)*Phi(z) + sigma*phi(z)
            # where z = (best - mu - xi)/sigma
            # If mu > best, z is negative, EI is small.
            # If mu < best, z is positive, EI is large.
            return ei

        # Phase 1: Initial Exploration
        # Use a small set of points to initialize the GP
        # We want to cover the space.
        n_init = 10
        if budget < n_init + 10:
            # Fallback to simple random search
            rng = np.random.default_rng(42)
            while calls_made < budget and time.time() < deadline:
                x = rng.uniform(-5, 5, dim)
                evaluate(x)
            return best_x, best_f

        # Generate initial points using a space-filling design (e.g., Halton or random)
        # Let's use random points for simplicity, but ensure spread.
        rng = np.random.default_rng(42)
        # Center
        evaluate(np.zeros(dim))
        # Axis extremes
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = 4.0
            p2 = np.zeros(dim); p2[i] = -4.0
            if calls_made < budget:
                evaluate(p1)
            if calls_made < budget:
                evaluate(p2)
        
        # Fill remaining init budget with random points
        while len(X) < n_init and calls_made < budget and time.time() < deadline:
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        # Phase 2: Bayesian Optimization Loop
        # Use EI acquisition
        # We will iterate until budget is exhausted or time is up.
        # To speed up GP computations, we can limit the number of points in the GP
        # or use a simplified kernel if it gets too slow.
        # For dim=5 and ~500 points, GP should be fast enough.
        
        length_scale = 1.5  # Initial guess
        noise = 1e-6
        
        # Adaptive length scale?
        # Let's keep it fixed for now, or adapt based on improvement.
        
        while calls_made < budget and time.time() < deadline:
            # Generate candidates
            # Use random sampling in the box
            n_candidates = 1000
            X_candidates = rng.uniform(-5, 5, (n_candidates, dim))
            
            # Predict
            mu, var = gp_predict(X_candidates, length_scale, noise)
            
            # Compute EI
            best_y = min(Y)
            ei = expected_improvement(mu, var, best_y, xi=0.0)
            
            # Select best candidate
            idx = np.argmax(ei)
            x_next = X_candidates[idx]
            
            # Evaluate
            f_next = evaluate(x_next)
            
            # Adaptive length scale: if we improved significantly, maybe reduce length scale (local)
            # If we didn't improve, maybe increase length scale (explore)
            # Simple heuristic:
            if f_next < best_y * 0.9: # 10% improvement
                length_scale *= 0.8
            else:
                length_scale *= 1.1
            
            # Clamp length scale
            length_scale = np.clip(length_scale, 0.1, 5.0)

        return best_x, best_f
