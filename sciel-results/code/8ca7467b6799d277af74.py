import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x_start, f_start, budget, rng, deadline):
        """Run CMA-ES for a given budget."""
        m = x_start.copy()
        sigma = 2.0
        C = np.eye(self.dim)
        
        lambda_pop = 4 + int(3 * np.log(self.dim))
        if lambda_pop > budget:
            lambda_pop = max(2, budget // 2) # Ensure at least some generations
            
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
        
        evals_used = 0
        f_best = f_start
        x_best = x_start.copy()
        
        while evals_used < budget and time.time() < deadline:
            # Determine how many to sample in this generation
            # Don't exceed budget
            n_sample = min(lambda_pop, budget - evals_used)
            if n_sample < 1:
                break
                
            # If we can't do a full generation, do a partial one?
            # CMA-ES assumes full generations for updates.
            # If remaining budget < lambda_pop, we just do random search or single step?
            # For simplicity, if remaining < lambda_pop, break and do random search?
            # Or just do one (1+1) step?
            # Let's just break if we can't do a full generation to keep CMA-ES valid.
            # Actually, we can just do a (1+1) step for the remainder.
            if n_sample < lambda_pop:
                 # Switch to (1+1)-ES for the remainder
                 while evals_used < budget and time.time() < deadline:
                     y = np.clip(m + sigma * rng.standard_normal(self.dim), -5, 5)
                     fy = func(y)
                     evals_used += 1
                     if fy < f_best:
                         f_best = fy
                         x_best = y.copy()
                         m = y.copy()
                         sigma *= 1.5
                     else:
                         sigma *= 0.87
                 break

            z = rng.standard_normal((n_sample, self.dim))
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(self.dim)
            
            y = z @ L.T
            x_pop = m + sigma * y
            x_pop = np.clip(x_pop, -5, 5)
            
            f_pop = np.array([func(x) for x in x_pop])
            evals_used += n_sample
            
            order = np.argsort(f_pop)
            x_sorted = x_pop[order]
            f_sorted = f_pop[order]
            
            if f_sorted[0] < f_best:
                f_best = f_sorted[0]
                x_best = x_sorted[0].copy()
                
            # Update mean
            m_new = np.sum(weights[:, np.newaxis] * x_sorted[:mu], axis=0)
            
            # Update path
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (m_new - m)
            
            step = (m_new - m) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * step
            
            # Sigma update
            # Check condition for h_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            # The denominator in the condition is sqrt(1 - (1-c_sigma)^(2 * age))
            # Age is roughly evals_used / lambda_pop
            age = evals_used / lambda_pop
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * age)) if age > 0 else 1.0
            if norm_p_sigma / denom > (1.4 + 2 / (self.dim + 1)) * chi_n:
                sigma /= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # h_sigma calculation
            h_sigma = 1.0 if norm_p_sigma / denom < (1.4 + 2 / (self.dim + 1)) * chi_n else 0.0
            
            # Covariance update
            # We need to be careful with the update if we used partial weights? No, we used full lambda.
            # But we might have used n_sample < lambda? No, we break if n_sample < lambda.
            
            # Rank one
            # Note: The term (1 - h_sigma) * c_c * (2 - c_c) * C is added to the rank-one update part?
            # Standard formula:
            # C = (1 - c_1 - c_mu + h_sigma * c_1 * (dim + 1)) * C
            #     + c_1 * (p_c p_c^T + (1 - h_sigma) * c_c * (2 - c_c) * C)
            #     + c_mu * sum(w_i (y_i - y_bar) (y_i - y_bar)^T)
            
            C = (1 - c_1 - c_mu + h_sigma * c_1 * (self.dim + 1)) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            # Rank mu
            # y_i are normalized steps: (x_i - m_old) / sigma
            y_mu = (x_sorted[:mu] - m) / sigma
            C += c_mu * np.sum(weights[:, np.newaxis]**2 * np.outer(y_mu.T, y_mu.T), axis=0)
            
            m = m_new
            C = (C + C.T) / 2

        return x_best, f_best

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)
        
        # Phase 1: Random Search
        n_rand = 40
        points = []
        for _ in range(n_rand):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            points.append((fx, x))
            
        if not points:
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            return x, fx
            
        points.sort(key=lambda p: p[0])
        
        # Phase 2: Portfolio of 2 CMA-ES
        # Budget allocation
        remaining = self.budget - len(points)
        budget1 = remaining // 2
        budget2 = remaining - budget1
        
        # Run 1
        x1, f1 = self._cma_es(func, points[0][1], points[0][0], budget1, rng, deadline)
        
        # Run 2 (if time and budget left)
        if time.time() < deadline and budget2 > 0:
            # Use second best random point
            x2, f2 = self._cma_es(func, points[1][1], points[1][0], budget2, rng, deadline)
            if f2 < f1:
                x1, f1 = x2, f2
                
        return x1, f1
