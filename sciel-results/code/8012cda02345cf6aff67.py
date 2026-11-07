"""Rewritten optimizer for BBOB f14 (Different powers) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer with restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 275  # Safety margin: 300s limit, 25s buffer
        rng = np.random.default_rng(42)
        d = self.dim
        max_calls = self.budget
        
        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            x = np.asarray(x).flatten()
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial sampling
        init_points = [np.zeros(d)]
        for _ in range(8):
            init_points.append(rng.uniform(-5, 5, d))
        
        for p in init_points:
            p_clipped = np.clip(p, -5, 5)
            fx = try_eval(p_clipped)
            if fx is None:
                break
        
        if best_x is None:
            best_x = np.zeros(d)
            best_f = try_eval(best_x)
            
        if calls_used >= max_calls or time.time() > deadline - 10:
            return best_x, best_f

        # Phase 2: Clean CMA-ES with adaptive restarts
        lambda_ = 9  # population size for d=5
        mu = lambda_ // 2  # selection pressure
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        max_restarts = 20
        min_sigma = 1e-7
        max_sigma = 8.0
        stagnation_threshold = 15  # generations without improvement
        
        for restart in range(max_restarts):
            if calls_used >= max_calls or time.time() > deadline - 5:
                break
            
            # Initialize CMA-ES state
            mean = best_x.copy()
            sigma = 1.0 if restart == 0 else max(0.5, 0.8 * sigma)
            C = np.eye(d)
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            
            # Cholesky factors
            try:
                L = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(L))
                B = L / D[:, np.newaxis]
            except np.linalg.LinAlgError:
                B = np.eye(d)
                D = np.ones(d)
            
            last_improvement = calls_used
            gen_count = 0
            
            # Main CMA-ES loop
            while calls_used < max_calls and time.time() < deadline - 3:
                gen_count += 1
                
                # Generate and evaluate population
                ys = np.array([mean + sigma * (B @ (D * rng.standard_normal(d))) for _ in range(lambda_)])
                ys = np.clip(ys, -5, 5)
                
                fvals = np.array([try_eval(y) for y in ys])
                
                if np.any(fvals is None) or calls_used >= max_calls:
                    break
                
                # Sort
                order = np.argsort(fvals)
                ys_sorted = ys[order]
                fvals_sorted = fvals[order]
                
                # Update best
                if fvals_sorted[0] < best_f:
                    best_f = fvals_sorted[0]
                    best_x = ys_sorted[0].copy()
                    last_improvement = calls_used
                
                # Select best mu
                y_selected = ys_sorted[:mu]
                
                # Update mean
                old_mean = mean.copy()
                mean = np.sum([w[i] * y_selected[i] for i in range(mu)], axis=0)
                mean = np.clip(mean, -5, 5)
                
                # Update evolution paths
                y_diff = (mean - old_mean) / sigma if sigma > 1e-10 else np.zeros(d)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ y_diff)
                
                h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen_count + 1))) < (1 + 2 / (d + 1)) * chi_n
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ y_diff)
                
                # Update covariance
                delta_hsig = (1 - h_sigma) * c_c * (2 - c_c)
                y_selected_diffs = (y_selected - old_mean) / sigma
                
                C = (1 - c_1 - c_mu + delta_hsig * c_1 * c_c * (2 - c_c)) * C
                C += c_1 * (np.outer(p_c, p_c) + delta_hsig * C)
                
                rank_mu = sum(w[i] * np.outer(y_selected_diffs[i], y_selected_diffs[i]) for i in range(mu))
                C += c_mu * rank_mu
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = max(min(sigma, max_sigma), min_sigma)
                
                # Update Cholesky
                try:
                    L = np.linalg.cholesky(C)
                    D = np.sqrt(np.diag(L))
                    B = L / D[:, np.newaxis]
                except np.linalg.LinAlgError:
                    C = np.eye(d)
                    B = np.eye(d)
                    D = np.ones(d)
                    sigma = max(sigma, 0.1)
                
                # Convergence checks
                if sigma < min_sigma or np.std(fvals_sorted) < 1e-12:
                    break
                
                # Stagnation
                if calls_used - last_improvement > stagnation_threshold * lambda_:
                    break

        # Phase 3: Local refinement with adaptive step size
        if calls_used < max_calls and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            min_sigma_local = 1e-10
            max_sigma_local = 0.5
            
            while calls_used < max_calls and time.time() < deadline - 1:
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, max_sigma_local)
                else:
                    sigma *= 0.7
                if sigma < min_sigma_local:
                    break
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
