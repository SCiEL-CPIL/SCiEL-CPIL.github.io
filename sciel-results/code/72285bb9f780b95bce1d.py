"""CMA-ES with restarts for BBOB f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with restarts for continuous black-box functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _run_cmaes(self, func, m_init, sigma_init, budget_remaining, start_time, deadline):
        """Run CMA-ES for a given budget, return (best_x, best_f, calls_used)."""
        n = self.dim
        lam = 8
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 2) / (mueff + 2) - 1) + c_sigma)
        c_c = (4 + mueff / n) / (n + 4 + mueff)
        c_mu = 2 * ((mueff - 2 + 2 / mueff) * c_c - 1) / (n + 2)
        
        m = m_init.copy()
        sigma = sigma_init
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = m.copy()
        best_f = float('inf')
        calls = 0
        gen = 0
        no_improve = 0
        best_f_prev = float('inf')
        
        while calls < budget_remaining:
            if time.time() - start_time > deadline:
                break
            
            # Check for stagnation: restart if sigma too small or no improvement
            if gen > 0 and (sigma < 1e-8 or (no_improve > 15 and gen > 10)):
                break
            
            # Sample population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)
                C = np.eye(n)
            
            X = np.zeros((lam, n))
            for i in range(lam):
                z = np.random.standard_normal(n)
                X[i] = m + sigma * (L @ z)
            
            # Check budget before evaluating
            if calls + lam > budget_remaining:
                # Only evaluate what's left
                n_eval = budget_remaining - calls
                X = X[:n_eval]
            else:
                n_eval = lam
            
            calls += n_eval
            
            # Evaluate
            F = np.array([func(x) for x in X])
            
            # Track global best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Sort
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update mean
            m_old = m.copy()
            # Simpler: just use the best if we have fewer than mu
            if n_eval >= mu:
                m = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            else:
                m = X_sorted[0].copy()
            
            # Update evolution paths
            y = (m - m_old) / sigma
            # Use generation count for h_sigma threshold
            chisq_n = np.sqrt(n) * (1 + 1/(4*n) - 1/(21*n**2))
            h_sigma = np.linalg.norm(y) / np.sqrt(1 - (1 - c_sigma)**(2*(gen+1))) < (1.4 + 2/(n+1)) * chisq_n
            h_sigma = 1.0 if h_sigma else 0.0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y
            
            # Update step size
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(n) - 1))
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_mu * (delta_h - 1)) * C
            for i in range(min(mu, n_eval)):
                Y_i = (X_sorted[i] - m_old) / sigma
                C += c_mu * weights[i] * np.outer(Y_i, Y_i)
            
            # Symmetrize and ensure PD
            C = (C + C.T) / 2
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-14:
                C = C + 1e-14 * np.eye(n)
            
            gen += 1
        
        return best_x, best_f, calls

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0
        
        best_x = np.zeros(n)
        best_f = float('inf')
        calls_used = 0
        
        # Phase 1: Initial exploration with larger sigma
        # Try a few different starting points
        rng = np.random.default_rng(42)
        
        # First, do a quick random search to find a decent starting point
        n_random = min(20, budget // 10)
        X_rand = rng.uniform(-5, 5, size=(n_random, n))
        F_rand = np.array([func(x) for x in X_rand])
        calls_used += n_random
        
        idx = np.argmin(F_rand)
        m_start = X_rand[idx].copy()
        best_x = m_start.copy()
        best_f = F_rand[idx]
        
        # Use the best random point as center, sigma based on spread
        sigma_start = max(1.0, np.std(F_rand) ** 0.25)
        
        # Run CMA-ES with restarts
        remaining = budget - calls_used
        m_init = m_start.copy()
        sigma_init = sigma_start
        
        while calls_used < budget:
            if time.time() - start_time > deadline:
                break
            
            bx, bf, c_used = self._run_cmaes(func, m_init, sigma_init, remaining, start_time, deadline)
            calls_used += c_used
            remaining = budget - calls_used
            
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
            
            # If CMA-ES didn't use much budget and stopped early, restart from best
            if c_used < remaining * 0.5 and c_used > 0:
                # Restart from the best found with a new sigma
                m_init = best_x.copy()
                sigma_init = max(0.5, sigma_init * 0.5)
            elif c_used == 0:
                break
            else:
                # CMA-ES ran to completion of its allocated budget portion
                # If we still have budget, do a smaller local refinement
                if remaining > 10:
                    m_init = best_x.copy()
                    sigma_init = 0.1
                else:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
