import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard
        start_time = time.time()
        deadline = start_time + 280  # Leave 20s margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        evals_used = 0
        
        def eval_point(x):
            nonlocal best_x, best_f, evals_used
            if evals_used >= budget:
                return best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global Initialization (10 points)
        # Sample 10 points uniformly in the box to find a good basin
        init_candidates = []
        for _ in range(10):
            x = rng.uniform(-5, 5, dim)
            f = eval_point(x)
            init_candidates.append((f, x))
        
        # Sort and pick best
        init_candidates.sort(key=lambda k: k[0])
        x_mean = init_candidates[0][1].copy()
        best_f = init_candidates[0][0]
        best_x = x_mean.copy()

        # Phase 2: CMA-ES (400 evaluations)
        # Parameters
        lam = 10
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        w = np.concatenate([w, np.zeros(lam - mu)])
        mueff = 1.0 / np.sum(w**2)
        
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 2) + c_c) / ((dim + 1.3) ** 2 + mueff + c_c)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        sigma = 1.0 # Moderate initial step size
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute L
        L = np.eye(dim)
        
        cma_budget = 400
        cma_evals = 0
        
        while cma_evals < cma_budget and evals_used < budget:
            if time.time() > deadline:
                break
                
            if budget - evals_used < lam:
                break

            # Generate offspring
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                if evals_used >= budget or time.time() > deadline:
                    break
                    
                z = rng.standard_normal(dim)
                y = L @ z
                x_new = x_mean + sigma * y
                f_val = eval_point(x_new)
                
                ys.append(y)
                xs.append(x_new)
                fs.append(f_val)
                
            if len(xs) < lam:
                break
                
            cma_evals += lam
            
            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)
            
            # Sort by fitness
            idx = np.argsort(fs)
            xs = xs[idx]
            ys = ys[idx]
            fs = fs[idx]
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:mu, None] * xs[:mu], axis=0)
            
            # Update C for next step's sampling
            # Calculate C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            # Update sigma
            norm_ps = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2))
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            
            # Update covariance
            sum_w_y = np.zeros((dim, dim))
            for i in range(mu):
                sum_w_y += w[i] * np.outer(ys[i], ys[i])
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * sum_w_y
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)
            
            # Update L
            L = np.linalg.cholesky(C)
            
            # Stop if converged
            if sigma < 1e-4:
                break

        # Phase 3: Local Refinement (Remaining budget)
        # Use coordinate-wise line search
        remaining = budget - evals_used
        if remaining > 0 and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            
            # Estimate local step size from sigma or fixed small value
            h = max(sigma * 0.5, 0.05)
            
            # We have ~100 evals left. 5 dims. 20 evals per dim.
            evals_per_dim = max(1, remaining // dim)
            
            for i in range(dim):
                if evals_used >= budget or time.time() > deadline:
                    break
                
                # Golden Section or Parabolic search along dimension i
                # Simple approach: Try moving +/- h, shrink h
                # More efficient: Parabolic interpolation
                x1 = x_local.copy()
                x2 = x_local.copy()
                x2[i] += h
                x3 = x_local.copy()
                x3[i] -= h
                
                f1 = eval_point(x1)
                f2 = eval_point(x2)
                f3 = eval_point(x3)
                
                # If we hit budget, stop
                if evals_used >= budget: break
                
                # Determine best direction
                if f2 < f1 and f2 < f3:
                    # Move right
                    # Do a few more steps to the right, shrinking h
                    for _ in range(evals_per_dim - 3):
                        if evals_used >= budget: break
                        x_try = x_local.copy()
                        x_try[i] += h
                        f_try = eval_point(x_try)
                        if f_try < f_local:
                            x_local = x_try
                            f_local = f_try
                            h *= 0.7
                        else:
                            # Try smaller step
                            x_try2 = x_local.copy()
                            x_try2[i] += h * 0.5
                            f_try2 = eval_point(x_try2)
                            if f_try2 < f_local:
                                x_local = x_try2
                                f_local = f_try2
                            h *= 0.5
                            if h < 1e-6: break
                            
                elif f3 < f1 and f3 < f2:
                    # Move left
                    for _ in range(evals_per_dim - 3):
                        if evals_used >= budget: break
                        x_try = x_local.copy()
                        x_try[i] -= h
                        f_try = eval_point(x_try)
                        if f_try < f_local:
                            x_local = x_try
                            f_local = f_try
                            h *= 0.7
                        else:
                            x_try2 = x_local.copy()
                            x_try2[i] -= h
                            f_try2 = eval_point(x_try2)
                            if f_try2 < f_local:
                                x_local = x_try2
                                f_local = f_try2
                            h *= 0.5
                            if h < 1e-6: break
                else:
                    # x1 is best, try smaller steps
                    h *= 0.5
                    for _ in range(evals_per_dim - 3):
                        if evals_used >= budget: break
                        x_try = x_local.copy()
                        x_try[i] += h
                        f_try = eval_point(x_try)
                        if f_try < f_local:
                            x_local = x_try
                            f_local = f_try
                            h *= 0.7
                        else:
                            x_try2 = x_local.copy()
                            x_try2[i] -= h
                            f_try2 = eval_point(x_try2)
                            if f_try2 < f_local:
                                x_local = x_try2
                                f_local = f_try2
                            h *= 0.5
                            if h < 1e-6: break

        return best_x, best_f
