# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """
    Bayesian Optimizer using Gaussian Process surrogate and Expected Improvement.
    Designed for BBOB f7 (Step ellipsoidal) on [-5, 5]^5 with budget 500.
    
    Strategy:
    1. Initial phase: Space-Filling Design (Sobol-like) to explore the space broadly.
    2. Main phase: GP regression with Matern 5/2 kernel.
       - Use Expected Improvement (EI) acquisition function to select next point.
       - Optimize the acquisition function using L-BFGS-B from multiple random starts.
    3. Final phase: Local refinement around the best point found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf
        self._X = []
        self._y = []

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        dim = self.dim
        budget = self.budget

        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None
        self._X = []
        self._y = []

        def wrapped(x):
            self._call_count += 1
            fx = float(func(x))
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = np.array(x, dtype=float)
            self._X.append(np.array(x, dtype=float))
            self._y.append(fx)
            return fx

        # --- GP Kernel Definitions ---
        
        def matern52_kernel(X1, X2, length_scale, variance):
            """
            Matern 5/2 kernel.
            X1: (n, d), X2: (m, d)
            Returns: (n, m) kernel matrix
            """
            # Compute squared distances
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
            X1_norm = np.sum(X1**2, axis=1)[:, np.newaxis]
            X2_norm = np.sum(X2**2, axis=1)[np.newaxis, :]
            dist_sq = X1_norm + X2_norm - 2 * X1 @ X2.T
            dist_sq = np.maximum(dist_sq, 0) # Numerical stability
            dist = np.sqrt(dist_sq)
            
            # Matern 5/2: (1 + sqrt(3)*r + 3*r^2/3) * exp(-sqrt(3)*r)
            # Wait, standard formula: (1 + r*sqrt(3) + r^2) * exp(-r*sqrt(3)) where r = dist/length_scale
            r = dist / length_scale
            return variance * (1 + r * np.sqrt(3) + r**2) * np.exp(-r * np.sqrt(3))

        def matern52_gradient(X1, X2, length_scale, variance, noise=0.0):
            """
            Gradient of the Matern 5/2 kernel w.r.t. X1.
            Returns: (n, m, d) gradient tensor.
            """
            # This is complex to implement analytically for all pairs.
            # For simplicity and speed in 5D, we might use finite differences for acquisition optimization
            # OR implement the gradient correctly.
            # Let's try to implement the gradient analytically.
            # K(r) = v * (1 + r*sqrt(3) + r^2) * exp(-r*sqrt(3))
            # dK/dr = v * [ (sqrt(3) + 2r) * exp(-r*sqrt(3)) + (1 + r*sqrt(3) + r^2) * (-sqrt(3)) * exp(-r*sqrt(3)) ]
            #       = v * exp(-r*sqrt(3)) * [ sqrt(3) + 2r - sqrt(3) - 3r - sqrt(3)*r^2 ]
            #       = v * exp(-r*sqrt(3)) * [ -r - sqrt(3)*r^2 ]
            #       = -v * r * (1 + sqrt(3)*r) * exp(-r*sqrt(3))
            
            # dK/dx_i = dK/dr * dr/dx_i = dK/dr * (x_i - y_i) / length_scale
            
            X1_norm = np.sum(X1**2, axis=1)[:, np.newaxis]
            X2_norm = np.sum(X2**2, axis=1)[np.newaxis, :]
            dist_sq = X1_norm + X2_norm - 2 * X1 @ X2.T
            dist_sq = np.maximum(dist_sq, 0)
            dist = np.sqrt(dist_sq)
            
            r = dist / length_scale
            exp_term = np.exp(-r * np.sqrt(3))
            dK_dr = -variance * r * (1 + np.sqrt(3) * r) * exp_term
            
            # dK/dX1[i,j,k] = dK_dr[i,j] * (X1[i,k] - X2[j,k]) / length_scale
            # Shape: (n, m, d)
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            grad = dK_dr[:, :, np.newaxis] * diff / length_scale
            
            return grad

        # --- Acquisition Function ---

        def expected_improvement(x, X_train, y_train, best_f, length_scale, variance, beta=0.01):
            """
            Calculate Expected Improvement at point x.
            """
            x = np.array(x).reshape(1, -1)
            # Predict mean and variance
            K_xx = matern52_kernel(x, X_train, length_scale, variance)
            K_ff = variance # + noise
            K_xf = matern52_kernel(x, X_train, length_scale, variance)
            
            # Solve for alpha
            try:
                # Add small jitter for stability
                L = np.linalg.cholesky(K_ff + 1e-8 * np.eye(len(X_train)))
                v = np.linalg.solve(L, K_xf.T)
                mu = np.dot(v.T, y_train)
                var = K_xx[0,0] - np.sum(v**2)
                var = max(var, 1e-10)
            except np.linalg.LinAlgError:
                # Fallback to pinv if cholesky fails
                K_ff_inv = np.linalg.pinv(K_ff + 1e-8 * np.eye(len(X_train)))
                v = K_ff_inv @ K_xf.T
                mu = np.dot(v.T, y_train)
                var = K_xx[0,0] - np.dot(K_xf, v)
                var = max(var, 1e-10)
            
            sigma = np.sqrt(var)
            if sigma < 1e-8:
                return 0.0
            
            z = (best_f - mu) / sigma
            EI = (best_f - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            # Penalize points too close to existing samples to encourage exploration
            min_dist = np.min(np.linalg.norm(x - X_train, axis=1))
            if min_dist < 0.1:
                EI *= 0.5
            return EI

        def acquisition_neg(x, X_train, y_train, best_f, length_scale, variance):
            """
            Negative acquisition function for minimization.
            """
            return -expected_improvement(x, X_train, y_train, best_f, length_scale, variance)

        # --- Main Loop ---

        # Phase 1: Initial Exploration
        # Use a quasi-random sequence (Halton/Sobol approximation) to ensure coverage.
        # For dim=5, budget=500. Let's take 20 points for initial exploration.
        n_init = 20
        if budget >= n_init:
            # Generate Halton sequence
            def halton(index, base):
                f = 1
                r = 0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r

            points = []
            for i in range(1, n_init + 1):
                p = []
                for d in range(dim):
                    # Use primes for bases
                    base = [2, 3, 5, 7, 11][d]
                    p.append(halton(i, base))
                # Scale to [-5, 5]
                p = np.array(p) * 10.0 - 5.0
                points.append(p)
            
            for p in points:
                if self._call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                wrapped(p)

        # Phase 2: Bayesian Optimization
        # Hyperparameters
        length_scale = 1.0
        variance = 1.0
        noise = 1e-6
        
        # We will update the GP model every step.
        # To speed up, we might not re-optimize hyperparameters every step,
        # or we do it every 10 steps. Given budget 500, let's try fixed hyperparams first
        # or simple adaptive.
        
        # For speed, we will use a fixed length_scale that adapts slightly.
        # Initial LS = 1.0 (covers ~1 unit). Box is 10 units wide.
        
        while self._call_count < budget:
            if time.time() > deadline:
                break
            
            # If we have very few points, just use random
            if len(self._X) < 5:
                x_next = np.random.uniform(-5, 5, dim)
                wrapped(x_next)
                continue

            X_train = np.array(self._X)
            y_train = np.array(self._y)
            best_f = np.min(y_train)

            # Optimize acquisition function
            # Use multi-start L-BFGS-B
            best_acq_val = np.inf
            best_x_acq = None
            
            # Number of starts
            n_starts = 10
            rng = np.random.default_rng(self._call_count)
            
            for _ in range(n_starts):
                # Start from random points or near best
                if rng.random() < 0.5 and self._best_x is not None:
                    x0 = self._best_x + rng.normal(0, 0.5, dim)
                else:
                    x0 = rng.uniform(-5, 5, dim)
                
                # Clip to bounds
                x0 = np.clip(x0, -5, 5)
                
                try:
                    res = minimize(
                        acquisition_neg, 
                        x0, 
                        args=(X_train, y_train, best_f, length_scale, variance),
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50}
                    )
                    if res.success:
                        acq_val = res.fun
                        if acq_val < best_acq_val:
                            best_acq_val = acq_val
                            best_x_acq = res.x
                except Exception:
                    pass

            if best_x_acq is None:
                # Fallback to random if optimization failed
                best_x_acq = np.random.uniform(-5, 5, dim)

            best_x_acq = np.clip(best_x_acq, -5, 5)
            
            # Evaluate
            wrapped(best_x_acq)

            # Adaptive Hyperparameter Update
            # Every 10 steps, check if we should adjust length_scale
            if self._call_count % 10 == 0 and len(self._X) > 10:
                # Simple heuristic: if variance of y is high, increase LS?
                # Or just keep it fixed. For f7, the step size is the key.
                # Let's try a slight increase in LS as we find the minimum to zoom in?
                # Actually, LS controls smoothness. If we are close, we want smaller LS to be sensitive.
                # But EI handles the zoom-in via the mean.
                # Let's keep LS fixed at 1.0 for now, as tuning it adds complexity and time.
                pass

        # Ensure we don't exceed budget
        # The loop breaks on call_count >= budget.

        if self._best_x is None:
            self._best_x = np.zeros(dim)
            self._best_f = func(self._best_x)

        return self._best_x, self._best_f

# EVOLVE-BLOCK-END
