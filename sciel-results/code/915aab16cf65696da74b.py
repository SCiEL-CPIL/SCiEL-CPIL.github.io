"""Alternative strategy: CMA-ES with adaptive population and restarts for BBOB f13."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard: 330s limit, leave 10s margin
        start_time = time.time()
        deadline = start_time + 320.0

        calls_used = 0
        best_x = None
        best_f = np.inf

        def eval_point(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= budget or time.time() > deadline:
                return np.inf
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES Parameters
        # Population size lambda
        # For dim=5, lambda is typically 4 + floor(3 * log(dim))
        # 4 + 3*1.6 = 8.8 -> 8. Let's use 10 for robustness.
        lambda_pop = 10
        
        # Initial step size
        sigma = 2.0
        
        # CMA-ES state
        mean = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Update rates
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initialize paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluation
        f_mean = eval_point(mean)
        
        # Main loop
        # We run CMA-ES until budget or time is exhausted
        # CMA-ES is memoryless in terms of calls per step (1 step = lambda calls)
        
        # To improve early progress (AOCC), we might want a smaller initial population or 
        # a hybrid approach, but pure CMA-ES is strong.
        # However, CMA-ES can be slow to start if sigma is too large.
        # Let's start with sigma=1.0 and adapt.
        
        # Reset sigma for better early behavior
        sigma = 1.0
        # Re-initialize C if we reset sigma? No, C is independent.
        
        # We need to handle the case where we restart if stuck.
        # For now, single run with adaptive sigma.
        
        while calls_used + lambda_pop <= budget and time.time() < deadline:
            # Generate population
            Y = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), size=lambda_pop)
            X = mean + sigma * (C_sqrt @ Y.T).T
            
            # Evaluate
            F = np.array([eval_point(x) for x in X])
            
            # Check if we ran out of budget mid-evaluation
            if np.any(np.isinf(F)):
                break
                
            # Sort by fitness
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            
            # Update mean (weighted average of top mu)
            mean_old = mean
            mean = np.sum(weights[:, None] * X[:mu], axis=0)
            
            # Update step size (sigma)
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C_sqrt^-1 * (mean - mean_old)
            # C_sqrt^-1 is the inverse of the square root. We can approximate or track it.
            # Standard CMA-ES updates C_sqrt via Cholesky or eigen decomposition.
            # For simplicity and speed in this context, we can use the eigen-decomposition update.
            
            # Calculate evolution path in distribution coordinates
            # We need C_sqrt_inv.
            # Let's use the standard update rules.
            
            # 1. Update p_sigma
            # We need C_sqrt_inv. 
            # Instead of tracking C_sqrt explicitly, we can update C and then factorize.
            # This is computationally heavier but more robust.
            
            # Let's switch to a simpler but effective variant: 
            # Update C using the standard formula, then Cholesky.
            
            # p_sigma update
            # Note: The term C_sqrt^-1 * (mean - mean_old) is the "natural gradient" step.
            # Let's compute the step in the standard space first.
            step = mean - mean_old
            
            # We need to map step to the distribution space.
            # This requires solving C_sqrt * z = step.
            # Since we are doing this every step, let's maintain the eigen decomposition.
            
            # Actually, for dim=5, Cholesky is fast.
            # Let's stick to the standard CMA-ES implementation structure.
            
            # 1. Update p_sigma
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ step
            
            # We need C_sqrt_inv. Let's compute it from C.
            # C is symmetric positive definite.
            # L = Cholesky(C)
            # C_sqrt_inv = inv(L)
            
            L = np.linalg.cholesky(C)
            # Solve L @ z = step for z
            z = np.linalg.solve(L, step)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
            
            # Check for sigma update
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_used // lambda_pop))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim + 1):
                sigma *= np.exp(c_sigma * d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(dim + 1) - 1))
                # Clip sigma to reasonable range
                sigma = np.clip(sigma, 1e-6, 10.0)
                
            # 2. Update p_c
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ step
            # C_inv_sqrt = inv(C_sqrt)
            # We have L, so C_sqrt = L.
            # C_inv_sqrt = inv(L)
            # Wait, C_sqrt in CMA-ES is usually the matrix such that C = C_sqrt @ C_sqrt.T
            # So C_sqrt = L.
            # C_inv_sqrt = inv(L)
            
            C_inv_sqrt = np.linalg.inv(L)
            step_dist = C_inv_sqrt @ step
            
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * step_dist
            
            # 3. Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c.T + (1 - (1 - c_c)**2) * C) + c_mu * sum(weights[i] * (x_i - mean_old) @ (x_i - mean_old).T)
            
            # Note: The rank-mu update uses the differences from the OLD mean.
            # Let's compute the weighted outer products.
            # x_i are the selected points.
            # diffs = X[:mu] - mean_old
            
            # Optimization: Only compute if c_mu > 0
            rank_mu_term = np.zeros((dim, dim))
            if c_mu > 0:
                diffs = X[:mu] - mean_old
                # rank_mu_term = sum(weights[i] * diffs[i] @ diffs[i].T)
                # Efficient computation:
                # rank_mu_term = (diffs * weights[:, None]).T @ diffs
                rank_mu_term = (diffs.T * weights) @ diffs
                
            # h_sigma term
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1 - c_sigma)^2 * t) < (1.4 + 2/(dim+1)) * sqrt(dim+1)
            # We already checked this for sigma update, but the condition for h_sigma is slightly different (threshold).
            # Standard threshold: (1.4 + 2/(dim+1)) * sqrt(dim+1)
            t = 1 - (1 - c_sigma)**(2 * (calls_used // lambda_pop))
            if t > 0:
                threshold = (1.4 + 2 / (dim + 1)) * np.sqrt(dim + 1)
                if np.linalg.norm(p_sigma) / np.sqrt(t) < threshold:
                    h_sigma = 1.0
                else:
                    h_sigma = 0.0
            else:
                h_sigma = 0.0
                
            # Update C
            C = (1 - c_1 - c_mu) * C 
            C += c_1 * (p_c @ p_c.T + (1 - h_sigma * (1 - (1 - c_c)**2)) * C)
            C += c_mu * rank_mu_term
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2.0
            # Add small regularization to ensure PD
            C += 1e-10 * np.eye(dim)
            
            # Re-cholesky for next step
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not PD, reset C
                C = np.eye(dim)
                L = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = 1.0
                mean = best_x if best_x is not None else np.zeros(dim)
                continue

        # If we still have budget, do a local polish with Nelder-Mead or simple gradient descent?
        # CMA-ES should have converged. 
        # If best_x is None, it means we failed.
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
