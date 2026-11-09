import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard: 300s limit, use 270s to be safe
        start_time = time.time()
        deadline = start_time + 270
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        evals_used = 0
        
        def eval_point(x):
            nonlocal best_x, best_f, evals_used
            if evals_used >= budget:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global Initialization
        # Sample points at different scales to locate the basin
        # f7 has a step, so we want to find the region where the function drops
        init_points = []
        # Center
        x0 = np.zeros(dim)
        f0 = eval_point(x0)
        init_points.append((f0, x0.copy()))
        
        # Random points in shrinking boxes
        scales = [4.5, 3.0, 1.5, 0.5]
        points_per_scale = 4
        for scale in scales:
            for _ in range(points_per_scale):
                if evals_used >= budget:
                    break
                x = rng.uniform(-scale, scale, dim)
                f = eval_point(x)
                init_points.append((f, x.copy()))
        
        # Sort and pick best
        init_points.sort(key=lambda k: k[0])
        x_start = init_points[0][1].copy()
        
        # Phase 2: CMA-ES
        # We will run CMA-ES with adaptive restarts if it stagnates
        
        def run_cma_es(mean_init, sigma_init, max_evals, seed_offset=0):
            nonlocal evals_used, best_x, best_f
            
            rng_local = np.random.default_rng(42 + seed_offset)
            
            # Population size
            lam = max(4, int(4 + 3 * np.log(dim)))
            mu = max(1, lam // 2)
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mueff = 1.0 / np.sum(w**2)
            
            # Learning rates
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = (2 - mueff / (dim + 2) + c_c) / ((dim + 1.3) ** 2 + mueff + c_c)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            
            sigma = sigma_init
            mean = mean_init.copy()
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Precompute Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
                
            cma_evals = 0
            best_in_run = np.inf
            no_improve_count = 0
            stagnation_window = 15
            restart_count = 0
            max_restarts = 3
            
            while cma_evals + lam <= max_evals and evals_used < budget:
                if time.time() > deadline:
                    break
                
                # Generate offspring
                ys = []
                xs = []
                fs = []
                
                for i in range(lam):
                    if evals_used >= budget or time.time() > deadline:
                        break
                    z = rng_local.standard_normal(dim)
                    y = L @ z
                    x_new = mean + sigma * y
                    f_val = eval_point(x_new)
                    
                    ys.append(y)
                    xs.append(x_new)
                    fs.append(f_val)
                    
                if len(xs) < lam:
                    break
                
                cma_evals += lam
                
                ys_arr = np.array(ys)
                xs_arr = np.array(xs)
                fs_arr = np.array(fs)
                
                # Sort by fitness
                idx = np.argsort(fs_arr)
                xs_sorted = xs_arr[idx]
                ys_sorted = ys_arr[idx]
                fs_sorted = fs_arr[idx]
                
                # Track improvement
                if fs_sorted[0] < best_in_run - 1e-10:
                    best_in_run = fs_sorted[0]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                # Update mean
                mean_old = mean.copy()
                mean = np.sum(w[:mu, None] * xs_sorted[:mu], axis=0)
                
                # Compute C_inv_sqrt for path updates
                # We need sqrt(C) inverse. Since C is SPD, we can use Cholesky
                # But L is sqrt(C) such that L L^T = C. So C^{-1/2} = L^{-T} L^{-1}
                # Actually, standard CMA-ES uses C^{-1/2} @ diff.
                # Let's compute eigendecomposition for stability or use Cholesky inverse
                try:
                    # C = L L^T. C^{-1/2} is not simply L^{-1}.
                    # We need matrix square root inverse.
                    # For small dim, eigendecomposition is fine.
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(dim)
                    
                # Update evolution paths
                diff = (mean - mean_old) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ diff)
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * (C_inv_sqrt @ diff)
                
                # Update sigma
                norm_ps = np.linalg.norm(p_sigma)
                chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2))
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                
                # Update covariance
                sum_w_y = np.zeros((dim, dim))
                for i in range(mu):
                    sum_w_y += w[i] * np.outer(ys_sorted[i], ys_sorted[i])
                    
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * sum_w_y
                C = (C + C.T) / 2
                C += 1e-12 * np.eye(dim)
                
                # Update L
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-10 * np.eye(dim)
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(dim)
                        L = np.eye(dim)
                        
                # Stop if converged
                if sigma < 1e-8:
                    break
                    
                # Stagnation-based restart
                if no_improve_count >= stagnation_window:
                    if restart_count < max_restarts and cma_evals + lam * 2 <= max_evals:
                        restart_count += 1
                        # Restart with larger sigma and reset covariance
                        sigma = max(sigma * 3.0, 0.5)
                        C = np.eye(dim)
                        p_sigma = np.zeros(dim)
                        p_c = np.zeros(dim)
                        L = np.eye(dim)
                        no_improve_count = 0
                        best_in_run = fs_sorted[0]
                    else:
                        break
                        
            return mean, sigma

        # Run first CMA-ES
        cma1_budget = min(350, budget - evals_used)
        if cma1_budget > 20:
            _, sigma_final = run_cma_es(x_start, 1.5, cma1_budget, seed_offset=0)
            
        # Phase 3: Second CMA-ES run if budget remains
        remaining = budget - evals_used
        if remaining > 20 and time.time() < deadline:
            # Start from best found, with smaller sigma for refinement
            sigma_restart = max(sigma_final * 0.5, 0.1)
            run_cma_es(best_x.copy(), sigma_restart, remaining, seed_offset=1000)
            
        return best_x, best_f
