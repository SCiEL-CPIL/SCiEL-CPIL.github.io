import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # Global state
        best_x = np.zeros(n)
        best_f = np.inf
        eval_count = 0
        
        class _BudgetExceeded(Exception):
            pass
        
        def safe_func(x):
            nonlocal best_x, best_f, eval_count
            if time.time() > deadline:
                raise _BudgetExceeded
            if eval_count >= budget:
                raise _BudgetExceeded
            f_val = func(np.array(x, dtype=np.float64))
            eval_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x, dtype=np.float64)
            return f_val

        rng = np.random.default_rng(42)
        
        # Phase 1: Structured initial sampling to find both global and local basins
        # Lunacek has a global min and a local min - we need to explore broadly
        n_init = min(40, budget)
        
        # Try origin first
        safe_func(np.zeros(n))
        
        # Structured grid-like exploration to cover the space
        # For 5D, use a combination of axis-aligned and diagonal points
        for i in range(1, n_init):
            if eval_count >= budget:
                break
            # Mix of random and structured points
            if i % 5 == 0:
                # Diagonal points at various scales
                scale = 2.0 * (i // 5 + 1) / 3.0
                x = np.full(n, scale * (1 if i % 10 == 0 else -1))
            else:
                # Random points with some structure
                x = rng.uniform(-5, 5, n)
                # Occasionally try points near known Lunacek features
                if i % 3 == 0:
                    x = rng.uniform(-3, 3, n)
            safe_func(x)
        
        if best_f == np.inf:
            return best_x, 0.0
        
        # Phase 2: CMA-ES with adaptive restarts
        x_mean = best_x.copy()
        sigma = 1.5  # Initial step size
        
        # CMA-ES parameters tuned for 5D
        lam = 8 * n  # 40 for dim=5 - more samples for better coverage
        mu = lam // 2  # 20
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        chi_n = np.sqrt(n) * (1 - 1/(4*n) - 1/(20*n**2))
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        ds = 1 + 2 * max(0, (mu_eff - 2) / (n + mu_eff)) + cs
        cmu = 2 * (mu_eff - 2 + 1/mu_eff) / ((n + 2)**2 + mu_eff)
        dsigma = 1.0
        
        # Covariance matrix
        C = np.eye(n)
        Cs = np.zeros(n)
        ps = np.zeros(n)
        
        def _cov_to_b_d(C_mat, n_dim):
            try:
                L = np.linalg.cholesky(C_mat + 1e-10 * np.eye(n_dim))
                return L, np.ones(n_dim)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.clip(eigvals, 1e-10, None)
                return eigvecs, np.sqrt(eigvals)
        
        B, D = _cov_to_b_d(C, n)
        
        no_improve_count = 0
        max_no_improve = 12
        restart_count = 0
        max_restarts = 5
        best_f_history = []
        
        while eval_count < budget and time.time() < deadline:
            # Generate lambda samples
            ys = []
            xs = []
            fs = []
            
            for k in range(lam):
                if time.time() > deadline:
                    break
                if eval_count >= budget:
                    break
                    
                z = rng.standard_normal(n)
                y = B @ (D * z)
                x = np.clip(x_mean + sigma * y, -5, 5)
                ys.append(y)
                xs.append(x)
                fs.append(safe_func(x))
                
            if eval_count >= budget or time.time() > deadline:
                break
            
            # Sort by fitness
            fs_array = np.array(fs)
            idx = np.argsort(fs_array)
            
            # Track improvement
            current_best = fs_array[idx[0]]
            if current_best < best_f - 1e-10:
                no_improve_count = 0
                best_f_history.append(best_f)
            else:
                no_improve_count += 1
            
            # If no improvement for too long, restart
            if no_improve_count > max_no_improve:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                
                # Adaptive restart strategy
                if restart_count % 2 == 1:
                    # Expand search
                    sigma *= 2.5
                else:
                    # Contract and recenter
                    sigma *= 0.4
                    x_mean = best_x.copy()
                
                no_improve_count = 0
                # Reset internal state partially
                ps = np.zeros(n)
                Cs = np.zeros(n)
                C = np.eye(n)
                B, D = _cov_to_b_d(C, n)
                continue
            
            # Calculate new mean
            w_sorted = w[:mu]
            selected_x = [xs[i] for i in idx[:mu]]
            new_mean = np.average(selected_x, axis=0, weights=w_sorted)
            
            # Calculate C^(-1/2) for ps update
            eigvals, eigvecs = np.linalg.eigh(C + 1e-10 * np.eye(n))
            eigvals = np.clip(eigvals, 1e-10, None)
            C_inv_sqrt = (eigvecs * (1/np.sqrt(eigvals))) @ eigvecs.T
            
            # Update ps
            ps = (1 - cs) * ps + cs * np.sqrt(mu_eff) * (C_inv_sqrt @ (new_mean - x_mean) / sigma)
            
            # Check hsig
            norm_ps = np.linalg.norm(ps)
            hsig = (norm_ps / chi_n - (1 - 0.5 * cs)) / (1 + cs) > 1.4
            hsig = 1.0 if hsig else 0.0
            
            # Update sigma
            sigma = sigma * np.exp((dsigma / ds) * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 0.005, 15.0)
            
            # Update Cs
            Cs = (1 - cs) * Cs + hsig * cs * ps
            
            # Update C
            y_sel = np.array([ys[idx[i]] for i in range(mu)])
            y_bar = np.average(y_sel, axis=0, weights=w_sorted)
            
            decay = 1 - cs * cs * (1 - ds) - cmu
            term_cs = cs * cs * (1 - ds) * np.outer(Cs, Cs)
            term_cmu = cmu * np.average([np.outer(y, y) for y in (y_sel - y_bar)], axis=0, weights=w_sorted)
            
            C = decay * C + term_cs + term_cmu
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = new_mean
            
            # Recompute B, D
            B, D = _cov_to_b_d(C, n)
            
            # If sigma is very small, likely converged
            if sigma < 0.005:
                break
        
        # Phase 3: Final local polish around best point
        remaining = budget - eval_count
        if remaining > 15 and time.time() < deadline - 5:
            polish_sigma = 0.3
            for _ in range(min(remaining // 3, 60)):
                if time.time() > deadline - 5 or eval_count >= budget:
                    break
                x = np.clip(best_x + rng.normal(0, polish_sigma, n), -5, 5)
                f_val = safe_func(x)
                polish_sigma *= 0.92
        
        return best_x, best_f
