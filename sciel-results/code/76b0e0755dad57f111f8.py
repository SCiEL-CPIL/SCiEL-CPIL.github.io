# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        used_calls = 0
        history = []

        def check_time():
            return time.time() > deadline

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                history.append((x.copy(), f))

        def count_func(x):
            nonlocal used_calls
            if used_calls >= budget:
                raise StopIteration
            used_calls += 1
            return func(x)

        # --- Phase 1: Coarse global exploration ---
        # Use Halton sequence for quasi-random exploration to cover space more efficiently
        n_global = 80
        for i in range(n_global):
            if check_time() or used_calls >= budget:
                break
            
            # Halton sequence for low-discrepancy sampling
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r
            
            x = np.zeros(d)
            for j in range(d):
                # Use different bases for each dimension (primes)
                base = [2, 3, 5, 7, 11][j % 5]
                x[j] = -5 + 10 * halton(i + 1, base)
            
            f = func(x)
            used_calls += 1
            try_update(x, f)
            history.append((x.copy(), f))

        if best_x is None:
            return None, np.inf

        # --- Phase 2: Bayesian Optimization with GP surrogate ---
        # Use remaining budget for BO iterations
        bo_budget = min(300, budget - used_calls)
        n_bo_iter = min(100, bo_budget)  # Each iteration uses 1-2 calls
        
        if n_bo_iter > 0 and best_x is not None:
            X = np.array([x for x, _ in history])
            Y = np.array([f for _, f in history])
            
            # Standardize Y for better GP performance
            y_mean = np.mean(Y)
            y_std = np.std(Y) + 1e-10
            Y_norm = (Y - y_mean) / y_std
            
            # GP hyperparameters (fixed for speed)
            length_scale = 1.0
            signal_variance = 1.0
            noise_variance = 1e-4
            
            for iter_i in range(n_bo_iter):
                if check_time() or used_calls >= budget:
                    break
                
                # Build GP kernel
                def kernel(X1, X2):
                    # RBF kernel
                    sq_dist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                    sq_dist = np.maximum(sq_dist, 0)  # numerical stability
                    return signal_variance * np.exp(-0.5 * sq_dist / (length_scale**2))
                
                K = kernel(X, X) + noise_variance * np.eye(len(X))
                
                # Solve for GP parameters
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
                except np.linalg.LinAlgError:
                    # Fallback to random exploration if GP fails
                    x_new = rng.uniform(-5, 5, d)
                    f_new = func(x_new)
                    used_calls += 1
                    try_update(x_new, f_new)
                    history.append((x_new.copy(), f_new))
                    X = np.vstack([X, x_new])
                    Y = np.vstack([Y, f_new])
                    Y_norm = (Y - y_mean) / y_std
                    continue
                
                # Expected Improvement acquisition function
                # Use EI with the best known value
                best_y_norm = np.min(Y_norm)
                
                def ei(x_sample):
                    # Predict mean and std at x_sample
                    k_star = kernel(x_sample.reshape(1, -1), X)[0]
                    mu = k_star @ alpha
                    v = np.linalg.solve(L, k_star)
                    sigma_sq = signal_variance - k_star @ v
                    sigma = np.sqrt(max(sigma_sq, 1e-10))
                    
                    # EI calculation
                    if sigma < 1e-10:
                        return 0.0
                    z = (mu - best_y_norm) / sigma
                    ei_val = (mu - best_y_norm) * norm.cdf(z) + sigma * norm.pdf(z)
                    return max(ei_val, 0)
                
                # Maximize EI using random restarts + local optimization
                n_restarts = 10
                best_ei = -np.inf
                best_x_ei = None
                
                for _ in range(n_restarts):
                    x_init = rng.uniform(-5, 5, d)
                    
                    def neg_ei(x):
                        return -ei(x)
                    
                    try:
                        res = scipy.optimize.minimize(
                            neg_ei, x_init, method='Nelder-Mead',
                            options={'maxiter': 20, 'xatol': 0.1, 'fatol': 1e-4}
                        )
                        if -res.fun > best_ei:
                            best_ei = -res.fun
                            best_x_ei = res.x
                    except Exception:
                        pass
                
                if best_x_ei is None:
                    best_x_ei = rng.uniform(-5, 5, d)
                
                best_x_ei = np.clip(best_x_ei, -5, 5)
                
                # Evaluate at best EI point
                f_new = func(best_x_ei)
                used_calls += 1
                try_update(best_x_ei, f_new)
                
                # Update GP training data
                X = np.vstack([X, best_x_ei])
                Y = np.vstack([Y, f_new])
                y_mean = np.mean(Y)
                y_std = np.std(Y) + 1e-10
                Y_norm = (Y - y_mean) / y_std

        # --- Phase 3: Local polish with CMA-ES ---
        # Final fine-tuning on the best found point
        if best_x is not None and used_calls < budget and not check_time():
            x = best_x.copy()
            f = best_f
            sigma = 0.05
            # Use remaining budget for polish
            max_polish_steps = min(50, budget - used_calls)
            for _ in range(max_polish_steps):
                if check_time() or used_calls >= budget:
                    break
                z = rng.standard_normal(d)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                used_calls += 1
                if fy < f:
                    x = y
                    f = fy
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                sigma = float(np.clip(sigma, 1e-6, 5.0))
                if sigma < 1e-5:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
