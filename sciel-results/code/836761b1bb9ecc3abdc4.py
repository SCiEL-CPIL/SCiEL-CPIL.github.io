import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc
from scipy.integrate import quad

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls = 0
        fbest = np.inf
        x_best = np.zeros(n)
        data = []  # List of (x, f) tuples

        def time_ok():
            return time.time() < deadline

        def eval(x):
            nonlocal calls, fbest, x_best, data
            if calls >= budget or not time_ok():
                return np.inf
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls += 1
            if fx < fbest:
                fbest = fx
                x_best = x.copy()
            data.append((x.copy(), fx))
            return fx

        # Phase 1: Initial Exploration
        # Use Sobol sampling for better coverage
        n_init = min(30, budget // 10)
        try:
            sampler = qmc.Sobol(d=n, scramble=True, seed=42)
            sobol_points = sampler.random(n_init)
            X_initial = sobol_points * 10.0 - 5.0
        except Exception:
            X_initial = rng.uniform(-5, 5, (n_init, n))
            
        for x in X_initial:
            if calls >= budget or not time_ok():
                break
            eval(x)
            
        # Also try origin
        if calls < budget and time_ok():
            eval(np.zeros(n))

        if not data:
            # Fallback
            x = rng.uniform(-5, 5, n)
            eval(x)

        # Phase 2: Bayesian Optimization (Gaussian Process + Expected Improvement)
        # We use a simple GP implementation.
        # Kernel: Matern 5/2
        # Acquisition: Expected Improvement (EI) with noise-free assumption (since func is deterministic)
        
        def matern_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
            """Matern 5/2 kernel."""
            dist = np.sqrt(np.sum((x1 - x2) ** 2, axis=-1))
            sqrt3_dist = np.sqrt(3) * dist / length_scale
            return sigma_f ** 2 * (1 + sqrt3_dist + (sqrt3_dist ** 2) / 3) * np.exp(-sqrt3_dist)

        def gp_predict(X_train, y_train, X_test, length_scale, sigma_f, noise=1e-8):
            """
            Compute mean and variance of GP at X_test given training data.
            Returns: mean (shape (N_test,)), var (shape (N_test,))
            """
            K = matern_kernel(X_train, X_train, length_scale, sigma_f)
            K += noise * np.eye(len(X_train))
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Fallback to adding jitter
                K += 1e-6 * np.eye(len(X_train))
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    # Return uniform prior if kernel is badly conditioned
                    mean = np.full(len(X_test), np.mean(y_train))
                    var = np.full(len(X_test), sigma_f ** 2)
                    return mean, var

            K_s = matern_kernel(X_test, X_train, length_scale, sigma_f)
            
            # Solve K alpha = y
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            
            # Mean
            mean = K_s @ alpha
            
            # Variance
            # v = L^{-1} K_s^T
            v = np.linalg.solve(L, K_s.T)
            var = sigma_f ** 2 - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 1e-10)  # Ensure positive variance
            
            return mean, var

        # Adaptive length scale estimation
        # Start with a reasonable guess
        current_length_scale = 1.5
        
        # BO Loop
        n_bo_iters = 30  # Number of BO iterations
        # We will use the remaining budget for BO, but we need to leave some for final refinement
        # Let's say we use 60% of remaining budget for BO, 40% for local refinement
        # But simpler: just run BO until budget is low or time is up.
        
        # We need to update the GP as we go.
        # To keep it fast, we will not re-optimize hyperparameters every step, 
        # but maybe adjust length_scale slowly or fix it. 
        # Given the time limit, fixed hyperparameters are safer.
        
        for _ in range(n_bo_iters):
            if calls >= budget - 20 or not time_ok() or (deadline - time.time()) < 30:
                break
                
            # Prepare training data
            X_train = np.array([d[0] for d in data])
            y_train = np.array([d[1] for d in data])
            
            if len(X_train) < 2:
                # Not enough data for GP, do random search
                x_try = rng.uniform(-5, 5, n)
                eval(x_try)
                continue
                
            # Evaluate EI on a candidate set
            # Generate a large set of candidate points (Latin Hypercube or Sobol)
            n_candidates = 1000
            try:
                cand_sampler = qmc.LatinHypercube(d=n, seed=rng)
                candidates = cand_sampler.random(n_candidates) * 10.0 - 5.0
            except Exception:
                candidates = rng.uniform(-5, 5, (n_candidates, n))
            
            # Predict mean and variance
            mean, var = gp_predict(X_train, y_train, candidates, current_length_scale, 1.0)
            std = np.sqrt(var)
            
            # Current best f
            f_min = np.min(y_train)
            
            # Expected Improvement
            # EI = (f_min - mean) * Phi(Z) + std * phi(Z)
            # where Z = (f_min - mean) / std
            # We want to minimize, so Improvement = f_min - f(x)
            # EI(x) = E[max(f_min - f(x), 0)]
            
            Z = (f_min - mean) / std
            # Phi: CDF of standard normal, phi: PDF
            from scipy.stats import norm
            Phi_Z = norm.cdf(Z)
            phi_Z = norm.pdf(Z)
            
            EI = (f_min - mean) * Phi_Z + std * phi_Z
            
            # Select best candidate
            idx = np.argmax(EI)
            x_next = candidates[idx]
            
            # Evaluate
            f_next = eval(x_next)
            
            # If no improvement for a while, consider adjusting length scale?
            # For now, keep it simple.
            
            # Optional: If stuck, do a random jump
            # Check if we improved
            if f_next >= fbest - 1e-10:
                # No improvement, maybe shrink search space or increase length scale?
                # Let's just continue.
                pass

        # Phase 3: Local Refinement with L-BFGS-B
        # Use the best point found so far
        if calls < budget and time_ok() and (deadline - time.time()) > 20:
            try:
                def lbfgs_func(x):
                    return eval(x)

                bounds = [(-5, 5)] * n
                
                # Run L-BFGS-B from the best point
                # Limit iterations to avoid running out of budget
                max_iter = min(50, budget - calls)
                
                res = minimize(
                    lbfgs_func,
                    x_best,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': max_iter,
                        'gtol': 1e-8,
                        'ftol': 1e-12,
                    }
                )
            except Exception:
                pass

        # Phase 4: Final fine-tuning with small perturbations
        if calls < budget and time_ok():
            for scale in [0.01, 0.005, 0.001, 0.0005]:
                if calls >= budget or not time_ok():
                    break
                for _ in range(min(5, budget - calls)):
                    if calls >= budget or not time_ok():
                        break
                    perturbation = rng.standard_normal(n) * scale
                    x_try = x_best + perturbation
                    x_try = np.clip(x_try, -5, 5)
                    f_try = eval(x_try)

        return x_best, fbest
