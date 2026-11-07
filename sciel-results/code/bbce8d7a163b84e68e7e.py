import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_used
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cma_es(x0, sigma0, max_calls):
            nonlocal best_x, best_f, calls_used
            n = dim
            # Population size
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights**2)
            
            # Strategy parameters
            cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
            
            # State
            m = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(20*n**2))
            
            # Main loop
            while calls_used < max_calls and time.time() < deadline:
                if calls_used + lam > budget:
                    # If we can't complete a full generation, break to avoid partial updates
                    # Or we could evaluate one by one, but CMA-ES relies on the full set.
                    # Let's break if we don't have enough budget for a full generation
                    # unless we are very close to the end.
                    if budget - calls_used < lam:
                        break

                # Generate population
                ys = []
                fs = []
                for i in range(lam):
                    if calls_used >= budget:
                        return
                    z = rng.standard_normal(n)
                    y = m + sigma * z
                    y = np.clip(y, -5, 5)
                    fy = evaluate(y)
                    ys.append(y)
                    fs.append(fy)
                
                # Sort by fitness
                order = np.argsort(fs)
                y_sorted = np.array(ys)[order]
                f_sorted = np.array(fs)[order]
                
                # Update mean
                m_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
                
                # Update covariance matrix
                y_diffs = (y_sorted[:mu] - m) / sigma
                
                # Check for convergence or stagnation
                # Simple convergence check: if step size is too small or variance is too low
                # We'll rely on the outer loop for restarts mostly.
                
                # Calculate h_sigma
                norm_ps = np.linalg.norm(ps)
                h_sig = 1.0 if norm_ps / chi_n < (1.4 + 2/(n+1)) else 0.0
                
                # Update pc
                pc = (1 - cs) * pc + h_sig * cs * mueff * y_diffs.mean(axis=0)
                
                # Update ps
                ps = (1 - cs) * ps + h_sig * np.sqrt(cs * (2 - cs) * mueff) * (m_new - m) / sigma
                
                # Update C
                # Rank-1 update
                c1_term = c1 * (pc * pc.T + (1 - (1 - cs)**2 * (1 + 2/mueff * (n - 1) if h_sig else 0)) * ps * ps.T)
                # Rank-mu update
                cmu_term = cmu * np.sum(weights[:, None, None] * np.outer(y_diffs, y_diffs), axis=0)
                
                C = (1 - c1 - cmu) * C + c1_term + cmu_term
                
                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2.0
                
                # Update sigma
                sigma *= np.exp(cs/5 * (np.linalg.norm(ps)/chi_n - 1))
                
                # Update mean
                m = m_new
                
                # Optional: check for convergence
                if sigma < 1e-10:
                    break

        # Phase 1: Initial CMA-ES from origin
        x0 = np.zeros(dim)
        cma_es(x0, 2.0, int(budget * 0.5))
        
        # Phase 2: If budget remains, do another CMA-ES from best found
        if calls_used < budget and best_x is not None:
            remaining = budget - calls_used
            if remaining > 20:
                # Restart with slightly larger sigma to escape local minima
                cma_es(best_x, 1.5, remaining)
        
        # Phase 3: Random restarts if budget allows
        remaining = budget - calls_used
        if remaining > 30:
            # Do a few random starts with small budgets
            n_starts = min(3, remaining // 20)
            for _ in range(n_starts):
                if calls_used >= budget:
                    break
                x_rand = rng.uniform(-4, 4, dim)
                # Give each start a small budget
                alloc = (budget - calls_used) // (n_starts - _ + 1)
                if alloc > 10:
                    cma_es(x_rand, 1.0, alloc)

        return best_x, best_f
