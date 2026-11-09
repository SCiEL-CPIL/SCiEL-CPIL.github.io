import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    """
    Bayesian Optimizer with Gaussian Process surrogate and Expected Improvement
    for BBOB f18 (Schaffers F7 ill-conditioned).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        calls = 0
        best_x = None
        best_f = np.inf
        X = []
        Y = []

        def eval_pt(x):
            nonlocal calls, best_x, best_f, X, Y
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            X.append(x.copy())
            Y.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol quasi-random global exploration
        n_init = min(15, budget)
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        init_points = sampler.random(n=n_init)
        init_points = 2.0 * init_points - 1.0
        init_points = 5.0 * init_points
        
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            eval_pt(init_points[i])
            
        if best_x is None:
            best_x = np.zeros(dim)
            eval_pt(best_x)

        # Phase 2: Bayesian Optimization Loop
        # GP Parameters
        nu = 2.5  # Matérn 5/2
        l = 1.0   # Length scale
        sigma_f = 1.0 # Signal variance
        sigma_n = 1e-6 # Noise variance (small for deterministic)
        
        # Precompute kernel helpers
        def matern_52(r):
            r = np.maximum(r, 1e-12)
            return np.exp(-np.sqrt(3) * r) * (1 + np.sqrt(3) * r + 3/5 * (np.sqrt(3) * r)**2)

        def compute_gp(X_obs, Y_obs, X_pred):
            """
            Computes mean and variance of GP at X_pred given X_obs, Y_obs.
            Returns mean (N_pred,), var (N_pred,)
            """
            X_obs = np.array(X_obs)
            Y_obs = np.array(Y_obs)
            X_pred = np.array(X_pred)
            
            n_obs = X_obs.shape[0]
            n_pred = X_pred.shape[0]
            
            # Compute K(X_obs, X_obs)
            K_obs = np.zeros((n_obs, n_obs))
            for i in range(n_obs):
                for j in range(i, n_obs):
                    dist = np.linalg.norm(X_obs[i] - X_obs[j]) / l
                    K_obs[i, j] = K_obs[j, i] = sigma_f**2 * matern_52(dist)
            K_obs += sigma_n**2 * np.eye(n_obs)
            
            # Compute K(X_pred, X_obs)
            K_pred_obs = np.zeros((n_pred, n_obs))
            for i in range(n_pred):
                for j in range(n_obs):
                    dist = np.linalg.norm(X_pred[i] - X_obs[j]) / l
                    K_pred_obs[i, j] = sigma_f**2 * matern_52(dist)
                    
            # Solve for alpha
            try:
                L = np.linalg.cholesky(K_obs)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_obs))
            except np.linalg.LinAlgError:
                # Fallback to lstsq if singular
                alpha = np.linalg.lstsq(K_obs, Y_obs, rcond=None)[0]
                
            # Mean
            mean = K_pred_obs @ alpha
            
            # Variance
            # Var = K(X_pred, X_pred) - K(X_pred, X_obs) K(X_obs, X_obs)^-1 K(X_obs, X_pred)
            # We only need variance for EI, so compute K(X_pred, X_pred)
            K_pred = np.zeros((n_pred, n_pred))
            for i in range(n_pred):
                for j in range(i, n_pred):
                    dist = np.linalg.norm(X_pred[i] - X_pred[j]) / l
                    K_pred[i, j] = K_pred[j, i] = sigma_f**2 * matern_52(dist)
                    
            # Compute cross term efficiently
            # V = K_pred - K_pred_obs @ inv(K_obs) @ K_pred_obs.T
            # Using Cholesky: inv(K_obs) = L^-T L^-1
            try:
                L_inv = np.linalg.inv(L)
                cross = K_pred_obs @ (L_inv.T @ (L_inv @ K_pred_obs.T))
                var = np.diag(K_pred) - np.sum(cross, axis=1)
            except:
                cross = K_pred_obs @ (np.linalg.solve(K_obs, K_pred_obs.T))
                var = np.diag(K_pred) - np.sum(cross, axis=1)
                
            var = np.maximum(var, 1e-12)
            return mean, var

        def expected_improvement(x_pred):
            """
            EI for minimization.
            """
            if len(X) == 0:
                return 0.0
            x_pred = np.array(x_pred).reshape(1, -1)
            mean, var = compute_gp(X, Y, x_pred)
            mu = mean[0]
            sigma = np.sqrt(var[0])
            if sigma < 1e-8:
                return 0.0
            improvement = best_f - mu
            z = improvement / sigma
            # Standard normal CDF and PDF
            from scipy.stats import norm
            EI = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            return -EI # Negate because we want to maximize EI, but minimize function

        # Acquisition function optimization
        # We will use a coarse grid to find a good starting point, then Nelder-Mead
        n_acq_grid = 20
        acq_sampler = qmc.Sobol(d=dim, scramble=True, seed=123)
        
        for _ in range(budget - calls):
            if calls >= budget or time.time() > deadline:
                break
            
            # Find best acquisition point
            best_acq_x = None
            best_acq_val = np.inf
            
            # Grid search for initialization of local optimizer
            grid_points = acq_sampler.random(n=n_acq_grid)
            grid_points = 2.0 * grid_points - 1.0
            grid_points = 5.0 * grid_points
            
            # Also include current best point neighborhood
            if best_x is not None:
                for r in [0.1, 0.5, 1.0]:
                    pert = best_x + r * np.random.randn(dim)
                    grid_points = np.vstack([grid_points, pert])
            
            # Evaluate EI on grid
            acq_vals = []
            for gp in grid_points:
                gp = np.clip(gp, -5.0, 5.0)
                try:
                    val = expected_improvement(gp)
                except:
                    val = 0.0
                acq_vals.append(val)
            
            acq_vals = np.array(acq_vals)
            best_grid_idx = np.argmin(acq_vals)
            start_x = grid_points[best_grid_idx]
            
            # Local optimization of EI using Nelder-Mead
            try:
                res = minimize(expected_improvement, start_x, method='Nelder-Mead', 
                               options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 50})
                if res.fun < best_acq_val:
                    best_acq_x = res.x
            except:
                best_acq_x = start_x
            
            best_acq_x = np.clip(best_acq_x, -5.0, 5.0)
            
            # Evaluate the selected point
            eval_pt(best_acq_x)
            
            # Update GP parameters adaptively? 
            # Simple heuristic: if improvement is small, increase exploration (increase length scale or noise)
            # For now, keep fixed for stability
            
        # Phase 3: Final local refinement with Nelder-Mead
        if calls < budget and best_x is not None and time.time() < deadline:
            # Use remaining budget for local polish
            # Estimate how many calls NM might take, cap it
            max_nm_calls = budget - calls
            if max_nm_calls > 0:
                # Wrap func to count calls and stop if budget exceeded
                nm_calls = [0]
                def nm_func(x):
                    nonlocal nm_calls
                    if nm_calls[0] >= max_nm_calls or time.time() > deadline:
                        return best_f
                    x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                    fx = float(func(x))
                    nm_calls[0] += 1
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    return fx

                try:
                    res = minimize(nm_func, best_x, method='Nelder-Mead', 
                                   options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': max_nm_calls})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5.0, 5.0)
                except Exception:
                    pass

        return best_x, best_f
