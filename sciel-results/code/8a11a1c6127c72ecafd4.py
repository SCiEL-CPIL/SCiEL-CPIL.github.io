import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Phase 1: Initial diverse sampling to find a good starting point
        n_random = 20
        candidates = [np.zeros(n)]
        candidates.append(np.ones(n) * 1.0)
        candidates.append(np.ones(n) * -1.0)
        candidates.append(np.ones(n) * 0.5)
        candidates.append(np.ones(n) * -0.5)
        rand_points = rng.uniform(-5, 5, (n_random, n))
        candidates.extend(rand_points)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        for x in candidates:
            if calls_used >= budget or time.time() > deadline:
                break
            try:
                f = func(x)
                calls_used += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            except Exception:
                break
        
        if best_x is None:
            return np.zeros(n), np.inf
        
        # Phase 2: CMA-ES optimization
        remaining = budget - calls_used
        if remaining <= 10:
            return best_x, best_f
        
        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(n)))  # Standard CMA-ES population size
        sigma = 2.0  # Initial step size (box is [-5,5]^n)
        mean = best_x.copy()
        C = np.eye(n)  # Covariance matrix
        C_inv = np.eye(n)
        step_size = sigma
        
        # CMA-ES adaptation parameters
        mu = pop_size // 2  # Number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # State
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Precompute Cholesky of C for sampling
        def cholesky_safe(M):
            try:
                L = np.linalg.cholesky(M)
                return L
            except:
                # If not positive definite, add small regularization
                eigvals, eigvecs = np.linalg.eigh(M)
                eigvals = np.maximum(eigvals, 1e-12)
                M_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
                return np.linalg.cholesky(M_reg)
        
        L = cholesky_safe(C)
        
        call_count = [0]
        
        def safe_func(x):
            if time.time() > deadline:
                raise ValueError("Time limit exceeded")
            if call_count[0] >= remaining:
                raise ValueError("Budget exceeded")
            call_count[0] += 1
            return func(x)
        
        try:
            for gen in range(500):  # Max generations
                if time.time() > deadline - 5 or call_count[0] >= remaining:
                    break
                
                # Generate population
                offspring = []
                for i in range(pop_size):
                    z = rng.standard_normal(n)
                    x = mean + step_size * (L @ z)
                    # Clip to box to ensure valid input
                    x = np.clip(x, -5.0, 5.0)
                    offspring.append(x)
                
                # Evaluate offspring
                offspring_f = []
                for x in offspring:
                    try:
                        f = safe_func(x)
                        offspring_f.append(f)
                    except Exception:
                        offspring_f.append(np.inf)
                
                # Sort by fitness
                indices = np.argsort(offspring_f)
                selected_x = [offspring[i] for i in indices[:mu]]
                selected_f = [offspring_f[i] for i in indices[:mu]]
                
                # Update best
                if selected_f[0] < best_f:
                    best_f = selected_f[0]
                    best_x = selected_x[0].copy()
                
                # Weighted mean of selected
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += weights[i] * selected_x[i]
                
                # Update sigma
                old_sigma = step_size
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv @ (new_mean - mean)) / step_size
                sigma_ratio = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))
                step_size *= np.exp(c_sigma / c_sigma * (sigma_ratio / np.sqrt(mu_eff) - 1))
                step_size = np.clip(step_size, 1e-12, 1e6)
                
                # Update covariance
                diff = (new_mean - mean) / step_size
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff
                delta_heav = (1 + (1 - c_c) * (c_c + 1) * (1 - c_sigma) / c_sigma) * np.sqrt(1 - (1 - c_c)**2) * mu_eff / (n + 2) - 1
                
                rank_1 = np.outer(diff, diff)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    xi = (selected_x[i] - mean) / step_size
                    rank_mu += weights[i] * np.outer(xi, xi)
                
                C = (1 + c_1 + c_mu - c_sigma) * C + c_1 * rank_1 + c_mu * rank_mu - c_sigma * (1 + delta_heav) * np.outer(p_c, p_c)
                
                # Ensure symmetry and positive semi-definiteness
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                L = cholesky_safe(C)
                
                mean = new_mean
                
                # Convergence check
                if step_size < 1e-10:
                    break
                
        except Exception:
            pass
        
        return best_x, best_f
