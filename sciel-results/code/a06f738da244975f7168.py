import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = None

    def __call__(self, func):
        start_time = time.time()
        self.deadline = start_time + 285  # Safety margin out of 300s
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            f = float(func(x_clipped))
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        def cma_es(x0, f0, max_iter, sigma_init=1.0):
            nonlocal best_x, best_f, calls
            
            n = dim
            lam = 4 + int(np.floor(np.log(n)))
            if lam < 2:
                lam = 2
            # For n=5, lam=5. Use slightly larger for stability
            lam = 9
            mu = lam // 2
            
            # Weights
            w = np.zeros(mu)
            for i in range(mu):
                w[i] = np.log(mu + 1) - np.log(i + 1)
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w**2)
            
            # Strategy parameters
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (mu_eff + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            # Initial state
            x_mean = x0.copy()
            sigma = sigma_init
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            hs = np.sqrt((1 + c_c) / c_c + 2) * (1 + c_sigma)
            
            for it in range(max_iter):
                if time.time() > self.deadline:
                    break
                if calls >= budget:
                    break
                
                # Generate offspring
                offspring = []
                for i in range(lam):
                    # Cholesky of C
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(n)
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = eval_func(y)
                    offspring.append((fy, y, z))
                
                # Sort by fitness
                offspring.sort(key=lambda x: x[0])
                
                # Update mean
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += w[i] * (offspring[i][1] - x_mean)
                x_mean = x_mean + y_w
                
                # Update sigma
                y_sigma = np.zeros(n)
                for i in range(mu):
                    y_sigma += w[i] * offspring[i][2]
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / hs - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Update C
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
                delta_hs = (1 + c_c) * (np.linalg.norm(pc) / hs - 1)
                C = (1 + c_1 * delta_hs - c_1 - c_mu) * C
                C += c_1 * np.outer(pc, pc)
                for i in range(mu):
                    C += c_mu * w[i] * np.outer(offspring[i][2], offspring[i][2])
                
                # Ensure C is symmetric and positive semi-definite
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    C = (C + C.T) / 2
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                
                # Update best
                if offspring[0][0] < best_f:
                    best_f = offspring[0][0]
                    best_x = offspring[0][1].copy()
                
                # Check for convergence
                if np.linalg.norm(y_w) < 1e-10 and sigma < 1e-6:
                    break
            
            return x_mean, best_f

        def one_plus_one_es(x0, fx0, max_iter):
            nonlocal best_x, best_f, calls
            
            x = x0.copy()
            fx = fx0
            sigma = 0.3
            no_improve = 0
            
            for _ in range(max_iter):
                if time.time() > self.deadline:
                    break
                if calls >= budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    no_improve = 0
                    sigma *= 1.15
                else:
                    no_improve += 1
                    sigma *= 0.8
                    if no_improve > 30:
                        break
                
                sigma = np.clip(sigma, 0.005, 5.0)
            
            return x, fx

        # Phase 1: Multi-start exploration
        n_starts = 6
        starts = [np.zeros(dim)]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))
        
        best_start_x = None
        best_start_f = np.inf
        
        for x0 in starts:
            if calls >= budget or time.time() > self.deadline:
                break
            x0 = np.clip(x0, -5, 5)
            f0 = eval_func(x0)
            if f0 < best_start_f:
                best_start_f = f0
                best_start_x = x0.copy()
        
        if best_start_x is None:
            best_start_x = np.zeros(dim)
            best_start_f = eval_func(best_start_x)
        
        # Phase 2: CMA-ES refinement from best start
        cma_budget = int((budget - calls) * 0.75)
        if cma_budget > 20:
            x_cma, f_cma = cma_es(best_start_x, best_start_f, cma_budget, sigma_init=1.0)
        
        # Phase 3: (1+1)-ES Polish
        es_budget = budget - calls
        if es_budget > 5:
            x_es, f_es = one_plus_one_es(best_x, best_f, es_budget)
        
        return best_x, best_f
