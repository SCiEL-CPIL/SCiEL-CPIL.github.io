"""Hybrid CMA-ES + Bayesian Optimization for Katsuura f23."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.stats import norm

class Optimizer:
    """
    Optimizes Katsuura f23 using a hybrid approach:
    1. Phase 1: CMA-ES with adaptive population size for robust global search.
    2. Phase 2: Bayesian Optimization with GP surrogate for fine-tuning near the optimum.
    3. Deadline guard ensures we return best result before timeout.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        lower = -5.0
        upper = 5.0
        
        calls_made = 0
        x_global_best = None
        f_global_best = np.inf
        
        def check_deadline():
            return time.time() > deadline
            
        def evaluate(x):
            nonlocal calls_made, x_global_best, f_global_best
            x_clipped = np.clip(x, lower, upper)
            fx = func(x_clipped)
            calls_made += 1
            if fx < f_global_best:
                f_global_best = fx
                x_global_best = x_clipped.copy()
            return x_clipped, fx

        # --- Phase 1: CMA-ES with Adaptive Population Size ---
        # Initial random start point
        x0 = rng.uniform(lower, upper, size=dim)
        x0, f0 = evaluate(x0)
        
        if calls_made >= budget or check_deadline():
            return x_global_best, f_global_best

        # CMA-ES state
        x_mean = x0.copy()
        sigma = 1.0  # Initial step size
        C = np.eye(dim)  # Covariance matrix
        p_sigma = np.zeros(dim)  # Cumulation for sigma control
        p_c = np.zeros(dim)  # Cumulation for covariance control
        
        # Control parameters
        mu = 2  # Number of recombination weights (top mu)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Population size schedule
        lam = 4  # Initial population size
        lam_max = 16  # Maximum population size
        lam_min = 4  # Minimum population size
        
        best_f = f0
        no_improve = 0
        max_no_improve = 10  # Generations without improvement before adapting lam
        
        # Track all evaluated points for potential BO initialization
        all_x = [x0.copy()]
        all_f = [f0]
        
        while calls_made < budget and not check_deadline():
            # Update mu to be half of lam, at least 2
            mu = max(2, lam // 2)
            # Recalculate weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # Generate population
            ys = []
            fs = []
            for i in range(lam):
                if calls_made >= budget or check_deadline():
                    break
                    
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset it
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (L @ z)
                y, fy = evaluate(y)
                ys.append(y)
                fs.append(fy)
                all_x.append(y.copy())
                all_f.append(fy)
            
            if len(ys) == 0:
                break
                
            # Sort by function value
            indices = np.argsort(fs)
            ys_sorted = [ys[i] for i in indices]
            fs_sorted = [fs[i] for i in indices]
            
            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                no_improve = 0
            else:
                no_improve += 1
                
            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += w[i] * ys_sorted[i]
            x_mean = y_w
            
            # Update sigma control
            y_sigma = (L.T @ (y_w / sigma))
            p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # Hessian potential
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * math.sqrt(mu_eff + 1 - 1)) else 0.0
            
            sigma *= math.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / math.sqrt(mu_eff) - 1))
            
            # Update covariance control
            y_c = (L.T @ (y_w / sigma))
            p_c = (1 - c_c) * p_c + math.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu + delta_h) * C
            C += c_1 * (np.outer(p_c, p_c) + delta_h * np.eye(dim))
            for i in range(mu):
                y_i = (L.T @ (ys_sorted[i] / sigma))
                C += c_mu * w[i] * np.outer(y_i, y_i)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-10:
                    C = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T
            except np.linalg.LinAlgError:
                C = np.eye(dim)
            
            # Adapt population size
            if no_improve >= max_no_improve:
                if lam < lam_max:
                    lam = min(lam * 2, lam_max)
                elif lam > lam_min:
                    lam = max(lam // 2, lam_min)
                no_improve = 0
                max_no_improve = min(max_no_improve * 2, 50)  # Increase patience
        
        # --- Phase 2: Bayesian Optimization for fine-tuning ---
        # Only enter BO if we have enough points and time remaining
        remaining_budget = budget - calls_made
        if remaining_budget > 20 and not check_deadline() and len(all_x) >= 10:
            # Prepare training data from all evaluated points
            X_train = np.array(all_x)
            y_train = np.array(all_f)
            
            # GP parameters
            length_scale = 0.5  # Smaller for fine-tuning
            noise = 1e-6
            signal_var = max(np.var(y_train), 1e-8)
            
            def rbf_kernel(X1, X2, length_scale, signal_var):
                """Compute RBF kernel matrix."""
                sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                sq_dists = np.maximum(sq_dists, 0)
                return signal_var * np.exp(-0.5 * sq_dists / length_scale**2)
            
            def gp_predict(X_test, X_train, y_train, length_scale, noise, signal_var):
                """Gaussian Process prediction: mean and variance."""
                K = rbf_kernel(X_train, X_train, length_scale, signal_var) + noise * np.eye(len(X_train))
                Ks = rbf_kernel(X_train, X_test, length_scale, signal_var)
                
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                    mu = Ks.T @ alpha
                    v = np.linalg.solve(L, Ks)
                    var = signal_var - np.sum(v**2, axis=0)
                    var = np.maximum(var, 1e-10)
                except np.linalg.LinAlgError:
                    # Fallback: simple interpolation
                    mu = np.full(X_test.shape[0], np.mean(y_train))
                    var = np.full(X_test.shape[0], np.var(y_train) + 1e-10)
                
                return mu, var
            
            def expected_improvement(mu, var, f_best, xi=0.01):
                """Compute Expected Improvement acquisition function."""
                sigma = np.sqrt(var)
                with np.errstate(divide='ignore', invalid='ignore'):
                    z = (f_best - mu - xi) / sigma
                    ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
                ei = np.where(sigma > 1e-10, ei, 0.0)
                return ei
            
            # Main BO loop
            bo_step = 0
            no_improve_count = 0
            f_best_prev = f_global_best
            
            while calls_made < budget and not check_deadline():
                bo_step += 1
                
                # Adaptive exploration parameter
                if bo_step < 10:
                    xi = 0.02  # More exploration early in BO phase
                elif bo_step < 30:
                    xi = 0.01
                else:
                    xi = 0.005  # More exploitation later
                
                # Generate candidate points for acquisition
                n_candidates = 300
                X_candidates = rng.uniform(lower, upper, size=(n_candidates, dim))
                
                # Add points near current best
                if x_global_best is not None:
                    n_near = 50
                    X_near = x_global_best + rng.normal(0, 0.3, size=(n_near, dim))
                    X_near = np.clip(X_near, lower, upper)
                    X_candidates = np.vstack([X_candidates, X_near])
                
                # Compute GP predictions
                mu, var = gp_predict(X_candidates, X_train, y_train, length_scale, noise, signal_var)
                
                # Compute EI
                ei = expected_improvement(mu, var, f_global_best, xi)
                
                # Select best candidate
                best_idx = np.argmax(ei)
                x_next = X_candidates[best_idx]
                x_next, f_next = evaluate(x_next)
                
                # Update training set
                X_train = np.vstack([X_train, x_next.reshape(1, -1)])
                y_train = np.append(y_train, f_next)
                
                # Track improvement
                if f_next < f_best_prev - 1e-8:
                    no_improve_count = 0
                    f_best_prev = f_next
                else:
                    no_improve_count += 1
                
                # Periodic reseeding to escape surrogate local optima
                if no_improve_count > 15 and calls_made < budget - 5:
                    # Inject random points
                    n_reseed = 3
                    for _ in range(n_reseed):
                        if calls_made >= budget or check_deadline():
                            break
                        x_rand = rng.uniform(lower, upper, size=dim)
                        x_rand, f_rand = evaluate(x_rand)
                        X_train = np.vstack([X_train, x_rand.reshape(1, -1)])
                        y_train = np.append(y_train, f_rand)
                    no_improve_count = 0

        return x_global_best, f_global_best
# EVOLVE-BLOCK-END
