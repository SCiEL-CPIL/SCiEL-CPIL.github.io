"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import warnings

class Optimizer:
    """Bayesian Optimization with Matern-5/2 GP and Local Refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # Data storage
        X = []
        y = []
        
        best_x = None
        best_f = np.inf
        
        def evaluate(x):
            nonlocal best_x, best_f
            x_eval = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x_eval))
            X.append(x_eval)
            y.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x_eval.copy()
            return fx

        # Phase 1: Initial Exploration (Latin Hypercube Sampling)
        # Weierstrass is multimodal. LHS ensures coverage.
        n_init = 50
        # Generate LHS points in [-5, 5]^dim
        # Simple LHS: divide each dimension into n_init bins, pick one random point per bin, shuffle
        X_lhs = np.zeros((n_init, dim))
        for d in range(dim):
            bins = np.linspace(-5, 5, n_init + 1)
            for i in range(n_init):
                X_lhs[i, d] = rng.uniform(bins[i], bins[i+1])
        # Shuffle rows to decorrelate
        idx = rng.permutation(n_init)
        X_lhs = X_lhs[idx]
        
        for x in X_lhs:
            if time.time() > deadline:
                break
            evaluate(x)
            
        if best_x is None:
            # Fallback if no evals happened (shouldn't happen)
            x0 = rng.uniform(-5, 5, dim)
            evaluate(x0)

        # Phase 2: Bayesian Optimization Loop
        # Use GP to guide next evaluations
        # We will switch to local refinement if budget is low or if we have enough points
        # Let's allocate ~300 evals to BO, rest to local refinement
        
        # GP Setup
        # Matern 5/2 is good for smooth but oscillatory functions like Weierstrass
        # Length scales: start with 1.0, let GP optimize? 
        # Optimizing hyperparameters at every step is expensive. 
        # We will fit the GP every step, but maybe cache or limit iterations.
        # For dim=5, n=50-350, fitting GP is O(n^3). 350^3 is ~40M ops. 
        # Python/NumPy overhead might make this slow.
        # Let's limit the number of BO steps to ensure we finish in time.
        # Or use a simpler surrogate if GP is too slow?
        # The directive suggests GP. Let's try it but with a tight loop.
        
        # To speed up GP fitting, we can limit the number of points used for training 
        # if it gets too large, but for 500 budget, n is small enough.
        
        # Acquisition Function: Expected Improvement (EI)
        def expected_improvement(X_candidate, y_mean, y_std):
            y_min = np.min(y)
            yi = y_mean
            ei = np.zeros(len(X_candidate))
            mask = yi > y_min
            if np.any(mask):
                delta = yi[mask] - y_min
                sigma = y_std[mask]
                # Avoid division by zero
                sigma = np.maximum(sigma, 1e-10)
                Z = delta / sigma
                # Standard normal CDF and PDF
                # Using scipy.stats for precision
                from scipy.stats import norm
                ei[mask] = delta * norm.cdf(Z) + sigma * norm.pdf(Z)
            return ei

        # Optimization for Acquisition Max
        # Since EI is non-differentiable (kink at best), we use multi-start local optimization
        # on the acquisition function.
        
        n_bo_steps = 200  # Leave 250 for local refinement
        calls_made = n_init
        
        for step in range(n_bo_steps):
            if time.time() > deadline:
                break
            if len(y) >= budget:
                break
                
            # Fit GP
            X_arr = np.array(X)
            y_arr = np.array(y)
            
            # Normalize y for numerical stability in GP?
            # GP handles scaling via ConstantKernel and WhiteKernel.
            
            # Kernel: Matern 5/2 with learned length scales
            # Initial length scales: 1.0
            # We might want to fix length scales to save time if optimization is slow?
            # Let's try optimizing them but with few iterations.
            
            try:
                kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-6)
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, normalize_y=True)
                # n_restarts_optimizer=1 to save time
                gp.fit(X_arr, y_arr)
            except Exception:
                # If GP fails, fall back to random sampling
                x_next = rng.uniform(-5, 5, dim)
                evaluate(x_next)
                continue

            # Maximize Acquisition
            best_acq_val = -np.inf
            best_x_acq = None
            
            # Multi-start for acquisition optimization
            n_starts = 10
            # Start points: best current points + random points
            start_points = []
            # Add best known points
            if len(X) > 0:
                # Take top 3 points
                y_sorted = np.argsort(y_arr)
                for i in y_sorted[:3]:
                    start_points.append(X_arr[i])
            # Add random points
            for _ in range(n_starts - 3):
                start_points.append(rng.uniform(-5, 5, dim))
            
            for sp in start_points:
                # Define objective for minimization (negative EI)
                def neg_ei(x):
                    x = np.clip(x, -5.0, 5.0)
                    y_mean, y_std = gp.predict(x.reshape(1, -1), return_std=True)
                    ei = expected_improvement(x.reshape(1, -1), y_mean[0], y_std[0])[0]
                    return -ei
                
                try:
                    res = minimize(neg_ei, sp, method='Nelder-Mead', 
                                   options={'maxiter': 50, 'xatol': 0.1, 'fatol': 1e-4})
                    if res.success:
                        x_cand = np.clip(res.x, -5.0, 5.0)
                        y_mean, y_std = gp.predict(x_cand.reshape(1, -1), return_std=True)
                        ei = expected_improvement(x_cand.reshape(1, -1), y_mean[0], y_std[0])[0]
                        if -ei > best_acq_val:
                            best_acq_val = -ei
                            best_x_acq = x_cand
                except Exception:
                    pass
            
            if best_x_acq is None:
                # Fallback to random
                best_x_acq = rng.uniform(-5, 5, dim)
                
            evaluate(best_x_acq)
            calls_made += 1
            
            # Check time more frequently
            if time.time() > deadline:
                break

        # Phase 3: Local Refinement
        # Use remaining budget for (1+1)-ES or Nelder-Mead from best_x
        # Nelder-Mead is good for smooth functions, but Weierstrass is non-smooth.
        # (1+1)-ES is robust. Let's use (1+1)-ES with adaptive sigma.
        
        if best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # Smaller initial sigma for refinement
            
            successes = 0
            total_steps = 0
            
            while calls_made < budget:
                if time.time() > deadline:
                    break
                    
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = float(func(y))
                calls_made += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    best_x = x.copy()
                    best_f = fx
                    successes += 1
                else:
                    pass
                
                total_steps += 1
                
                if total_steps % 10 == 0:
                    success_rate = successes / total_steps
                    if success_rate > 0.2:
                        sigma *= 1.3
                    else:
                        sigma *= 0.7
                    successes = 0
                    total_steps = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
