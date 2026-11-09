"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for Katsuura function optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, budget, deadline, rng, best_state):
        """Run CMA-ES for a given budget, updating best_state if improved."""
        n = self.dim
        x = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        mu = max(1, n // 2)
        nu = 1 / (2 * n + 3)
        cmu = 2 / ((n + 4)**2 - mu)
        cnu = (mu + 2) / ((n + 2)**2 + mu)
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        damps = 1 + 2 * max(0, np.sqrt((mu - 1)/(n + 1)) - 1) + cmu
        dampc = (4 + mu/n) / (n + 4 + 2*mu/n)
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        pc_norm = chiN * (1 + 2/(n+1) + 1/(n+1))
        
        calls_used = 0
        
        for gen in range(budget // mu + 1):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break
            
            # Generate lambda samples
            lam = max(2 * mu + 1, min(4 * mu + 2, budget - calls_used))
            B = np.linalg.cholesky(C)
            samples = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                if calls_used >= budget:
                    break
                z = rng.standard_normal(n)
                y = x + sigma * B @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                samples.append((fy, y))
            
            if not samples:
                break
            
            # Sort by fitness
            samples.sort(key=lambda s: s[0])
            
            # Update mean
            w = np.log(np.arange(1, mu + 1)) - np.mean(np.log(np.arange(1, mu + 1)))
            w = w / np.sum(w)
            x_new = sum(w[i] * samples[i][1] for i in range(mu))
            
            # Update step size control
            y_mean = (x_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chiN * y_mean
            
            # Hessian drift control
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**((gen + 1) / c_sigma)) > (1.4 + 1/n) * chiN:
                sigma *= np.exp(0.3 * c_sigma * (np.linalg.norm(p_sigma)/chiN - 1))
            
            # Update evolution path for covariance
            p_c = (1 - c_c) * p_c + c_c * (pc_norm / sigma) * y_mean
            
            # Update covariance matrix
            delta_d = (1 + c_c * (2 - c_c)) * (1 if np.dot(p_c, p_c) / (pc_norm**2) < (1.4 + 2/(n+1)) * chiN**2 else 0)
            rank_mu = sum(w[i] * np.outer(samples[i][1] - x, samples[i][1] - x) for i in range(mu))
            C = (1 + cmu * (mu_eff - 1) - cmu - c_c * (2 - c_c)) * C \
                + cmu * (rank_mu + (1 - delta_d + cmu * (mu_eff - 1)) * np.outer(p_c, p_c)) \
                + c_c * (2 - c_c) * (1 - delta_d) * np.outer(p_c, p_c)
            
            # Symmetrize and ensure positive definiteness
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update best state
            if samples[0][0] < best_state[1]:
                best_state[0] = samples[0][1].copy()
                best_state[1] = samples[0][0]
            
            # Check for convergence
            if sigma < 1e-10 or np.all(np.abs(x_new - x) < 1e-10):
                break
            
            x = x_new
        
        return calls_used, best_state

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        
        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(0)
        
        best_x = np.zeros(n)
        best_fx = np.inf
        best_state = [best_x, best_fx]
        
        # Phase 1: Quick scan to find a good region
        scan_budget = min(20, total_budget // 10)
        for i in range(scan_budget):
            if time.time() > deadline:
                break
            x = rng.uniform(-4, 4, n)
            fx = func(x)
            if fx < best_state[1]:
                best_state[0] = x.copy()
                best_state[1] = fx
        
        # Phase 2: CMA-ES from the best found point
        remaining_budget = total_budget - scan_budget
        sigma0 = 2.0
        calls_used, best_state = self._cmaes(func, best_state[0], sigma0, remaining_budget, deadline, rng, best_state)
        
        # Phase 3: If we have budget left, do a few more restarts
        remaining = total_budget - calls_used - scan_budget
        if remaining > 30 and time.time() < deadline - 5:
            n_restarts = min(3, remaining // 40)
            for r in range(n_restarts):
                if time.time() > deadline:
                    break
                x0 = rng.uniform(-4, 4, n)
                # Perturb from current best
                if np.random.rand() < 0.5:
                    x0 = best_state[0] + 1.5 * rng.standard_normal(n)
                    x0 = np.clip(x0, -4, 4)
                budget_r = min(remaining // (n_restarts - r), 80)
                calls_used_r, best_state = self._cmaes(func, x0, 1.5, budget_r, deadline, rng, best_state)
                remaining -= budget_r
        
        return best_state[0], best_state[1]
# EVOLVE-BLOCK-END
