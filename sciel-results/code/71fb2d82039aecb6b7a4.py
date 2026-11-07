import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        deadline = start_time + 280  # safety margin before 300s limit

        # Multi-start CMA-ES with progressively smaller initial sigmas
        # For Sphere (convex), larger sigma finds the basin, smaller refines
        sigmas = [4.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05]
        
        calls = 0
        best_x = np.zeros(n)
        best_f = float('inf')
        
        # Track best per-start for potential reuse
        for sigma_init in sigmas:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            
            # Run CMA-ES with this initial sigma
            result_x, result_f, calls_used = self._run_cmaes(
                func, n, budget - calls, sigma_init, deadline
            )
            calls += calls_used
            
            if result_f < best_f:
                best_f = result_f
                best_x = result_x.copy()
            
            # If we've converged well (relative to our best), we can stop
            # For Sphere, if we're within 1e-4 of what we think is optimal, 
            # further refinement has diminishing returns for AOCC
            if best_f < 1e-4:
                break
        
        return best_x, best_f

    def _run_cmaes(self, func, n, budget, sigma_init, deadline):
        """Run CMA-ES for up to budget evaluations, respecting deadline."""
        # CMA-ES Parameters
        lambda_pop = 4 + int(3 * np.log(n))
        mu = lambda_pop // 2
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Initialization
        x_m = np.zeros(n)
        sigma = sigma_init
        
        C = np.eye(n)
        D = 1.0
        B = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        calls = 0
        best_x = x_m.copy()
        best_f = float('inf')
        
        # Stagnation tracking
        last_improve_call = 0
        stagnation_window = 50  # calls without improvement before considering restart
        
        while calls < budget:
            # Time check
            if time.time() > deadline:
                break
            
            # Stagnation check: if no improvement for stagnation_window calls, break
            # This allows the outer loop to try a different sigma
            if calls > 0 and calls - last_improve_call > stagnation_window and sigma < 1.0:
                # Only restart if we've already tried smaller sigmas
                # For the first large sigma, let it run longer
                break
            
            # 1. Sample population
            Z = np.random.randn(lambda_pop, n)
            Y = (B.T @ Z.T).T  # (lambda, n)
            X = x_m + D * sigma * Y
            F = np.zeros(lambda_pop)
            
            # Evaluate all points
            for i in range(lambda_pop):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                x_i = X[i]
                F[i] = func(x_i)
                calls += 1
                
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x_i.copy()
                    last_improve_call = calls
            
            if calls >= budget or time.time() > deadline:
                break
            
            # Sort population
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            Z_sorted = Z[order]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()
                last_improve_call = calls
                
            # 2. Update mean
            x_m_old = x_m.copy()
            x_m = np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # 3. Update sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Avoid division by zero in early iterations
            if calls > lambda_pop:
                denom = np.sqrt(1 - (1 - c_sigma)**(2 * (calls - lambda_pop)))
            else:
                denom = 1.0
            threshold = (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff + 1) / (mu_eff - 1))
            
            if norm_p_sigma / denom < threshold:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
                
            y = np.sum(w[:, np.newaxis] * Z_sorted[:mu], axis=0)
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ y)
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # 4. Update C (Covariance)
            norm_p_c = np.linalg.norm(p_c)
            if calls > lambda_pop:
                denom_c = np.sqrt(1 - (1 - c_c)**(2 * (calls - lambda_pop)))
            else:
                denom_c = 1.0
            threshold_c = (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff + 1) / (mu_eff - 1))
            
            if norm_p_c / denom_c < threshold_c:
                h_c = 1.0
            else:
                h_c = 0.0
                
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ y)
            
            Z_w = w[:, np.newaxis] * Z_sorted[:mu]
            rank_mu_term = Z_w.T @ Z_w
            
            C = (1 + c_1 * (h_c - 1) - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_c + c_1) * np.eye(n)) + c_mu * rank_mu_term
            C = (C + C.T) / 2
            
            eigenvalues, B = np.linalg.eigh(C)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            D = np.sqrt(eigenvalues)
            
            # Stagnation: if D is too small, break
            if D[0] < 1e-6:
                break
        
        return best_x, best_f, calls
