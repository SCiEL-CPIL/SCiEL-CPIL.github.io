import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        N = self.budget
        
        start_time = time.time()
        deadline = start_time + 280
        
        best_x_global = None
        best_f_global = np.inf
        
        def run_cmaes(x0, sigma0, budget_remaining):
            nonlocal best_x_global, best_f_global, deadline, start_time
            
            lambda_ = max(4, int(4 + 3 * np.log(d)))
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            c_1 = 2 / ((d + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
            
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
            
            calls = 0
            best_x = x_mean.copy()
            best_f = func(best_x)
            calls += 1
            best_x_global = best_x.copy()
            best_f_global = best_f
            
            gen = 0
            no_improve_count = 0
            stagnation_threshold = 15
            
            while calls + lambda_ <= budget_remaining and time.time() < deadline:
                gen += 1
                
                # Check time before expensive ops
                if time.time() > deadline - 5:
                    break
                
                # Sample using Cholesky of C
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(d))
                except np.linalg.LinAlgError:
                    # Fallback to identity if C is not PD
                    L = np.eye(d)
                
                # Sample population
                pop = np.zeros((lambda_, d))
                for i in range(lambda_):
                    z = np.random.randn(d)
                    pop[i] = x_mean + sigma * (L @ z)
                    pop[i] = np.clip(pop[i], -5, 5)
                
                # Evaluate population
                f_pop = np.array([func(x) for x in pop])
                calls += lambda_
                
                # Sort by fitness
                order = np.argsort(f_pop)
                pop = pop[order]
                f_pop = f_pop[order]
                
                # Update best
                if f_pop[0] < best_f:
                    best_f = f_pop[0]
                    best_x = pop[0].copy()
                    best_x_global = best_x.copy()
                    best_f_global = best_f
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Check for stagnation and restart
                if no_improve_count >= stagnation_threshold:
                    break
                
                # Update mean
                x_mean_new = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
                
                # Update ps
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (L.T @ (x_mean_new - x_mean))
                
                # Update pc
                h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * gen)) < (1 + 2 / (d + 1)) * chi_n)
                pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma
                
                # Update C
                diff = pop[:mu] - x_mean
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C \
                    + c_1 * np.outer(pc, pc) \
                    + c_mu / sigma**2 * np.einsum('i,ij,ik->jk', weights, diff, diff)
                
                # Symmetrize C
                C = (C + C.T) / 2
                
                # Update sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update mean
                x_mean = x_mean_new
                
                # Convergence check
                if sigma < 1e-12 or np.all(np.abs(C - np.trace(C) / d * np.eye(d)) < 1e-10):
                    break
            
            return calls, best_f_global
        
        # Phase 1: Initial exploration with moderate sigma
        x0 = np.zeros(d)
        f0 = func(x0)
        best_x_global = x0.copy()
        best_f_global = f0
        
        # Try a few random starting points to find a good initial region
        for _ in range(min(10, N // 10)):
            if time.time() > deadline:
                break
            x_try = np.random.uniform(-4, 4, d)
            f_try = func(x_try)
            if f_try < best_f_global:
                best_f_global = f_try
                best_x_global = x_try.copy()
        
        # Phase 2: Run CMA-ES from best point found
        calls_used = 1 + min(10, N // 10)
        remaining = N - calls_used
        if remaining > lambda_ + 1:
            run_cmaes(best_x_global, 2.0, remaining)
        
        # Phase 3: If still have budget, do local refinement
        if time.time() < deadline - 10 and best_f_global < np.inf:
            # Simple local search around best point
            x_local = best_x_global.copy()
            f_local = best_f_global
            sigma_local = 0.1
            for _ in range(50):
                if time.time() > deadline:
                    break
                if f_local >= np.inf:
                    break
                y = np.clip(x_local + sigma_local * np.random.randn(d), -5, 5)
                fy = func(y)
                if fy < f_local:
                    x_local, f_local = y, fy
                    sigma_local *= 1.2
                    if f_local < best_f_global:
                        best_f_global = f_local
                        best_x_global = x_local.copy()
                else:
                    sigma_local *= 0.8
                if sigma_local < 1e-8:
                    break
        
        return None
