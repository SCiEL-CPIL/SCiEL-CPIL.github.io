"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive restarts for robust global optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = float('inf')
        
        # Phase 1: Initial random sampling to find a good starting point
        init_budget = max(10, int(total_budget * 0.15))
        
        # Evaluate origin as a strong prior for Schaffers F7
        x0 = np.zeros(n)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        
        for _ in range(init_budget - 1):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Multi-restart CMA-ES
        remaining_budget = total_budget - init_budget
        max_restarts = 3
        budget_per_restart = max(50, remaining_budget // max_restarts)
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            if restart == 0:
                x = best_x.copy()
                fx = best_f
                sigma = 1.0
            else:
                # New restart: sample around current best with some perturbation
                x = best_x + 0.5 * rng.standard_normal(n)
                x = np.clip(x, -5, 5)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma = 1.5
            
            # Simple CMA-ES-like strategy with population
            pop_size = max(4, int(1 + 3 * np.log(n)))
            lambda_ = pop_size
            mu = max(1, lambda_ // 2)
            c_c = (4 + lambda_ / n) / ((n + 4) + (lambda_ / n))
            c_1 = 2 / ((n + 1.3) ** 2 + lambda_)
            c_mu = min(1 - c_1, 2 * ((mu + 0.5) / (n + 2) ** 2 - c_1))
            
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            C_eig = np.linalg.eigh(C)
            C_eigvals = C_eig[0]
            C_eigvecs = C_eig[1]
            
            # Precompute inverse square root of C for sampling
            def sample_from_C():
                z = rng.standard_normal(n)
                # C = V diag(w) V^T, sample as V diag(sqrt(w)) z
                return C_eigvecs @ (np.sqrt(np.maximum(C_eigvals, 1e-10)) * z)
            
            iter_count = 0
            no_improve = 0
            max_iter_no_improve = 15
            
            while iter_count < budget_per_restart:
                if time.time() > deadline:
                    break
                
                # Generate population
                ys = []
                for i in range(lambda_):
                    y = x + sigma * sample_from_C()
                    y = np.clip(y, -5, 5)
                    ys.append(y)
                
                # Evaluate population
                fvals = []
                for y in ys:
                    if time.time() > deadline:
                        fvals.append(best_f + 1000)
                    else:
                        fvals.append(func(y))
                    iter_count += 1
                
                # Sort by fitness
                sorted_idx = np.argsort(fvals)
                ys_sorted = [ys[i] for i in sorted_idx]
                fvals_sorted = [fvals[i] for i in sorted_idx]
                
                # Update best
                if fvals_sorted[0] < best_f:
                    best_f = fvals_sorted[0]
                    best_x = ys_sorted[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1
                
                # Update mean
                w = np.log(np.arange(1, mu + 1))
                w /= w.sum()
                mu_eff = 1.0 / np.sum(w ** 2)
                
                old_x = x.copy()
                x = sum(w[i] * ys_sorted[i] for i in range(mu))
                
                # Update step size (CSA-like)
                d_norm = np.linalg.norm(x - old_x)
                if d_norm > 1e-10:
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff / lambda_) * (x - old_x) / (sigma * np.sqrt(np.trace(C)))
                    chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
                    sigma *= np.exp((c_sigma/5) * (np.linalg.norm(p_sigma)/chi_n - 1))
                    sigma = np.clip(sigma, 1e-8, 10)
                
                # Update path and covariance
                if d_norm > 1e-10:
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff / lambda_) * (x - old_x) / (sigma * np.sqrt(np.trace(C)))
                
                delta_hs = (1 + 2 * max(0, (np.linalg.norm(p_c)**2 / n) - 1) / (1 - c_c)) - 1
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + delta_hs * np.eye(n)) + c_mu * sum(w[i] * np.outer(ys_sorted[i] - old_x, ys_sorted[i] - old_x) for i in range(mu))
                
                # Periodic eigendecomposition to ensure positive definiteness
                if iter_count % 10 == 0:
                    C_eig = np.linalg.eigh(C)
                    C_eigvals = np.maximum(C_eig[0], 1e-10)
                    C_eigvecs = C_eig[1]
                    C = C_eigvecs @ np.diag(C_eigvals) @ C_eigvecs.T
                
                # Check for stagnation -> break to next restart
                if no_improve >= max_iter_no_improve:
                    break
            
            # If we used all budget, stop
            if iter_count >= budget_per_restart:
                if time.time() > deadline:
                    break
                # Continue with next restart if budget remains
                remaining_budget = total_budget - init_budget - iter_count
                if remaining_budget < 50:
                    break
                budget_per_restart = remaining_budget
        
        return best_x, best_f
# EVOLVE-BLOCK-END
