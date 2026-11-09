"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        call_count = 0
        
        # Wrapper to count calls and enforce budget
        def counted_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return best_f
            call_count += 1
            try:
                f_val = func(x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return 1e100

        # 1. Initial Exploration: Sobol sequence for space-filling
        n_init = 20
        X_init = np.random.uniform(-5, 5, (n_init, dim))
        # Force one point at 0,0,0,0,0 as Rosenbrock often has global min nearby after rotation/shift
        X_init[0] = np.zeros(dim)
        
        Y_init = np.array([counted_func(x) for x in X_init])
        
        # Filter out any failures (if func threw exception, Y might be 1e100)
        mask = Y_init < 1e99
        X_data = X_init[mask]
        Y_data = Y_init[mask]
        
        # 2. Bayesian Optimization Loop
        # Kernel: Matern with nu=2.5 is good for smooth functions like Rosenbrock
        # Length scales initialized to cover the box roughly
        kernel = Matern(length_scale=np.ones(dim), length_scale_bounds=(0.1, 10.0), nu=2.5)
        
        # We will run BO until budget is nearly exhausted or time is up
        # Each iteration: 
        # 1. Fit GP
        # 2. Optimize Acquisition (EI)
        # 3. Evaluate best point
        
        while call_count < budget - 5 and time.time() < deadline:
            if len(X_data) == 0:
                break
                
            try:
                # Fit Gaussian Process
                # Normalize Y to help GP convergence
                Y_mean = np.mean(Y_data)
                Y_std = np.std(Y_data)
                if Y_std < 1e-10:
                    Y_std = 1.0
                Y_norm = (Y_data - Y_mean) / Y_std
                
                gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, alpha=1e-10, n_restarts_optimizer=2)
                gp.fit(X_data, Y_norm)
                
                # Define Expected Improvement (EI) acquisition function
                def ei(x):
                    x = np.array(x).reshape(1, -1)
                    mu, sigma = gp.predict(x, return_std=True)
                    # Convert back to original scale for EI calculation
                    mu_orig = mu * Y_std + Y_mean
                    sigma_orig = sigma * Y_std
                    
                    current_best = np.min(Y_data)
                    
                    if sigma_orig[0] < 1e-10:
                        return -(mu_orig[0] - current_best)
                    
                    imp = current_best - mu_orig[0]
                    z = imp / sigma_orig[0]
                    # Standard Normal CDF
                    from scipy.stats import norm
                    ei_val = imp * norm.cdf(z) + sigma_orig[0] * norm.pdf(z)
                    return -ei_val[0] # Negate for minimization
                
                # Optimize the acquisition function
                # Use multiple random starts to find the global max of EI
                # We have a small budget left for these local optimizations
                # We can afford ~5-10 local minimizations of EI per BO step
                
                n_acq_starts = 5
                if call_count > budget - 20:
                    n_acq_starts = 1
                    
                best_acq_x = None
                best_acq_val = float('inf')
                
                rng = np.random.default_rng(call_count)
                for _ in range(n_acq_starts):
                    x0 = rng.uniform(-5, 5, dim)
                    try:
                        res = minimize(ei, x0, method='L-BFGS-B', bounds=[(-5, 5)]*dim, options={'maxiter': 50})
                        if res.fun < best_acq_val:
                            best_acq_val = res.fun
                            best_acq_x = res.x
                    except Exception:
                        pass
                
                if best_acq_x is None:
                    # Fallback to random
                    best_acq_x = rng.uniform(-5, 5, dim)
                    
                # Evaluate the new point
                f_new = counted_func(best_acq_x)
                
                # Add to data
                X_data = np.vstack([X_data, best_acq_x])
                Y_data = np.append(Y_data, f_new)
                
            except Exception:
                # If GP fitting fails, just do a random step
                if call_count < budget:
                    x_rand = np.random.uniform(-5, 5, dim)
                    f_rand = counted_func(x_rand)
                    X_data = np.vstack([X_data, x_rand])
                    Y_data = np.append(Y_data, f_rand)

        return best_x, best_f

# EVOLVE-BLOCK-END
