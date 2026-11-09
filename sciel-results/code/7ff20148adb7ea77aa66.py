import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 280.0
        
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters - use smaller lambda for more generations
        # With 500 budget, lambda=8 gives ~62 generations, lambda=10 gives ~50
        lam = 8
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
        
        # Initialization - reduced sigma for faster early convergence
        x_mean = np.zeros(dim)
        sigma = 1.5
        
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        evals_used = 0
        
        def cholesky_safe(C_mat):
            try:
                return np.linalg.cholesky(C_mat)
            except np.linalg.LinAlgError:
                C_mat = C_mat + 1e-6 * np.eye(dim)
                return np.linalg.cholesky(C_mat)
        
        L = cholesky_safe(C)
        
        def compute_c_inv_sqrt(C_mat, L_mat):
            """Compute C^{-1/2} using Cholesky: C = LL^T, so C^{-1} = L^{-T}L^{-1}
            C^{-1/2} = (C^{-1})^{1/2}. For 5x5, eigendecomposition is fine but
            we can also use: if C = LL^T, then C^{-1/2} = L^{-1} (L^{-T}L^{-1})^{1/2}
            Simpler: just use eigendecomposition for the 5x5 case - it's fast enough."""
            eigvals, eigvecs = np.linalg.eigh(C_mat)
            eigvals = np.clip(eigvals, 1e-12, None)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        C_inv_sqrt = compute_c_inv_sqrt(C, L)
        
        # Initial evaluation
        f_mean = func(x_mean)
        evals_used += 1
        best_f = f_mean
        best_x = x_mean.copy()
        
        prev_best_f = best_f
        stall_count = 0
        stall_threshold = 5  # generations without improvement
        
        while evals_used < budget:
            if time.time() > deadline:
                break
            
            if budget - evals_used < lam:
                break
            
            # Check for stall - if no improvement for several generations, 
            # reduce sigma and continue or break to local refinement
            if best_f < prev_best_f - 1e-10:
                stall_count = 0
            else:
                stall_count += 1
            
            prev_best_f = best_f
            
            if stall_count > stall_threshold and sigma < 0.1:
                break
            
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                if evals_used >= budget or time.time() > deadline:
                    break
                    
                z = rng.standard_normal(dim)
                y = L @ z
                x_new = x_mean + sigma * y
                x_new = np.clip(x_new, -5, 5)
                
                f_val = func(x_new)
                evals_used += 1
                
                ys.append(y)
                xs.append(x_new)
                fs.append(f_val)
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_new.copy()
                    
            if len(xs) < lam:
                break
                
            ys_arr = np.array(ys)
            xs_arr = np.array(xs)
            fs_arr = np.array(fs)
            
            idx = np.argsort(fs_arr)
            xs_sel = xs_arr[idx]
            ys_sel = ys_arr[idx]
            
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:mu, None] * xs_sel[:mu], axis=0)
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            norm_ps = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2))
            
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            sum_w_y = np.zeros((dim, dim))
            for i in range(mu):
                sum_w_y += w[i] * np.outer(ys_sel[i], ys_sel[i])
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * sum_w_y
            
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)
            
            L = cholesky_safe(C)
            C_inv_sqrt = compute_c_inv_sqrt(C, L)
            
            # If sigma is very small, break to local refinement
            if sigma < 1e-5:
                break
        
        # Local Refinement using (1+1)-ES with CMA-ES geometry
        remaining = budget - evals_used
        if remaining > 0 and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            
            # Use a small sigma for local search, scaled by the CMA-ES sigma
            sigma_local = max(sigma * 0.5, 0.01)
            
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                
                z = rng.standard_normal(dim)
                y = L @ z
                x_new = x_local + sigma_local * y
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                evals_used += 1
                
                if f_new < f_local:
                    f_local = f_new
                    x_local = x_new.copy()
                    best_f = f_local
                    best_x = x_local.copy()
                    sigma_local *= 1.1
                else:
                    sigma_local *= 0.9
                    
                sigma_local = np.clip(sigma_local, 1e-8, 1.0)

        return best_x, best_f
