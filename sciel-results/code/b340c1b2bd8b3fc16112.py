import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0 # 20s safety margin

        best_x = None
        best_f = np.inf
        calls_made = [0]

        def counted_func(x):
            calls_made[0] += 1
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            return float(func(x))

        # 1. Initial Exploration
        # Use a space-filling design or random points.
        # 10 points is a good start for d=5.
        n_init = 10
        X = np.random.uniform(-4.5, 4.5, size=(n_init, dim))
        Y = np.array([counted_func(x) for x in X])
        
        # Update best
        idx_min = np.argmin(Y)
        best_f = Y[idx_min]
        best_x = X[idx_min].copy()

        # 2. Gaussian Process Setup
        # We use a Matern 5/2 kernel with fixed hyperparameters.
        # This avoids the cost of hyperparameter optimization.
        # Lengthscale: 1.5 (allows correlation over ~1-2 units)
        # Signal Variance: 100 (covers typical range of f24)
        # Noise Variance: 1e-6 (numerical stability)
        
        lengthscale = 1.5
        signal_var = 100.0
        noise_var = 1e-6

        def matern52_kernel(X1, X2, ls, sv):
            # X1: (n1, d), X2: (n2, d)
            # Compute squared distances
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1.dot(X2.T)
            sq_dists = np.maximum(sq_dists, 0)
            dists = np.sqrt(sq_dists) / ls
            # Matern 5/2
            sqrt5_r = np.sqrt(5.0) * dists
            K = sv * (1.0 + sqrt5_r + (5.0/3.0) * dists**2) * np.exp(-sqrt5_r)
            return K

        # 3. Bayesian Optimization Loop
        n_bo = budget - n_init
        # We will do BO for most of the budget, leaving ~50 for polish
        n_polish = 50
        n_bo_actual = max(0, n_bo - n_polish)
        
        # Pre-compute Cholesky factorization for speed? 
        # No, X grows. We re-fit each time.
        # To speed up, we can cache the Cholesky of the previous K and update?
        # Incremental Cholesky is complex. With N=500, full re-fit is ~0.01-0.05s per step.
        # 450 steps * 0.05s = 22.5s. This is well within budget.

        for i in range(n_bo_actual):
            if calls_made[0] >= budget or time.time() > deadline:
                break

            # Fit GP
            K = matern52_kernel(X, X, lengthscale, signal_var)
            K += noise_var * np.eye(len(X))
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # If Cholesky fails, add jitter
                K += 1e-5 * np.eye(len(X))
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    # Skip this step if numerical issues persist
                    continue

            # Solve for alpha
            # L * alpha = Y  => alpha = L^{-1} Y
            # Using solve_triangular
            L_inv_Y = np.linalg.solve_triangular(L, Y, lower=True)
            alpha = np.linalg.solve_triangular(L.T, L_inv_Y, lower=False)

            # Define Acquisition Function (Expected Improvement)
            # We want to minimize f.
            # EI(x) = (mu_min - mu(x) - xi) * Phi(Z) + sigma(x) * phi(Z)
            # where Z = (mu_min - mu(x) - xi) / sigma(x)
            # mu_min is the current best function value.
            mu_min = best_f
            xi = 0.05 # Exploration parameter
            
            def acquisition(x):
                x = x.reshape(1, -1)
                K_s = matern52_kernel(x, X, lengthscale, signal_var)
                # K_s shape: (1, N)
                
                # Predictive mean
                mu = K_s.dot(alpha)
                
                # Predictive variance
                v = np.linalg.solve_triangular(L, K_s.T, lower=True)
                sigma2 = signal_var - np.sum(v**2, axis=0)
                sigma = np.sqrt(np.maximum(sigma2, 1e-10))
                
                # EI
                if sigma < 1e-8:
                    return 0.0
                
                z = (mu_min - mu - xi) / sigma
                ei = (mu_min - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
                
                return -ei # We want to maximize EI, so minimize -EI

            # Maximize Acquisition
            # Use multi-start Nelder-Mead on the acquisition function
            n_starts = 10
            best_acq_val = -np.inf
            best_acq_x = None
            
            rng = np.random.default_rng(42 + i)
            for _ in range(n_starts):
                # Start near best known point or random?
                # Mix: 50% near best, 50% random
                if rng.random() < 0.5:
                    x0 = best_x + rng.normal(0, 1.0, dim)
                else:
                    x0 = rng.uniform(-4.8, 4.8, dim)
                
                x0 = np.clip(x0, -5, 5)
                
                # Simple local optimization using Nelder-Mead
                # Since we don't have scipy.optimize.minimize easily accessible in a tight loop without overhead,
                # we can use a simple coordinate descent or a few NM steps.
                # However, scipy.optimize is available.
                from scipy.optimize import minimize
                
                res = minimize(acquisition, x0, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
                
                if res.fun < best_acq_val:
                    best_acq_val = res.fun
                    best_acq_x = res.x

            if best_acq_x is None:
                continue

            # Clip and evaluate
            x_new = np.clip(best_acq_x, -5, 5)
            f_new = counted_func(x_new)
            
            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

            # Update Dataset
            X = np.vstack([X, x_new])
            Y = np.append(Y, f_new)

        # 4. Polish Phase
        # Use Nelder-Mead on the true function starting from best_x
        if calls_made[0] < budget and time.time() < deadline and best_x is not None:
            from scipy.optimize import minimize
            
            def obj(x):
                if calls_made[0] >= budget or time.time() > deadline:
                    return np.inf
                return counted_func(x)
            
            # Simple NM polish
            x0 = best_x
            simplex = [x0.copy()]
            spread = 0.5
            for d in range(dim):
                v = x0.copy()
                v[d] += spread
                simplex.append(v)
            
            # Manual NM to avoid overhead of scipy.optimize.minimize if it's too slow?
            # Scipy NM is fine.
            res = minimize(obj, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x

        if best_x is None:
            best_x = np.zeros(dim)
            if calls_made[0] < budget:
                best_f = counted_func(best_x)

        return best_x, best_f
