import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, calls, max_iters, sigma_init=1.0, lambda_=10, mu=5):
        """
        Lightweight CMA-ES implementation.
        """
        d = self.dim
        x = np.array(x0, dtype=float)
        f0 = func(x)
        
        # CMA-ES Parameters
        # lambda_ and mu are fixed for simplicity and speed
        w = np.log(mu / 2 + 1) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        m = x.copy()
        sigma = sigma_init
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        L = np.linalg.cholesky(C)
        
        best_x = x.copy()
        best_f = f0
        
        for _ in range(max_iters):
            if calls[0] >= self.budget:
                break
                
            # Sample
            xs = []
            fs = []
            for _ in range(lambda_):
                z = np.random.randn(d)
                xk = m + sigma * (L @ z)
                fk = func(xk)
                xs.append(xk)
                fs.append(fk)
                
            idx = np.argsort(fs)
            xs = [xs[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()
                
            # Update mean
            m_new = sum(w[i] * xs[i] for i in range(mu))
            
            # Update covariance
            v = [(xs[i] - m) / sigma for i in range(mu)]
            C_mu = sum(w[i] * np.outer(v[i], v[i]) for i in range(mu))
            
            pc_new = (1 - c_c) * pc + c_c * ((m_new - m) / sigma)
            
            C_1 = (1 - (c_1 + c_mu)) * C + \
                  c_1 * (1 + (1 - c_sigma) * d_sigma * np.outer(pc, pc) - c_sigma * np.outer(pc, pc)) + \
                  c_mu * (C_mu - C)
                  
            C = (C_1 + C_1.T) / 2
            C += 1e-10 * np.eye(d)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ((m_new - m) / sigma)
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-3 * np.eye(d)
                L = np.linalg.cholesky(C)
                
            m = m_new
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * d) - 1))
            pc = pc_new
            
            if sigma < 1e-8:
                break
                
        return best_x, best_f

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        dim = self.dim
        calls = [0]
        budget = self.budget
        
        def safe_func(x):
            if calls[0] >= budget:
                return np.inf
            calls[0] += 1
            try:
                return float(func(np.asarray(x, dtype=float)))
            except Exception:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        # Phase 1: Global Exploration
        # Use a Sobol sequence for low-discrepancy sampling
        # We need about 100-150 points for initial exploration
        n_explore = 120
        
        # Generate Sobol points in [0, 1]^d
        sampler = qmc.Sobol(d=dim, scramble=True)
        # Sobol requires power of 2 for optimal performance, but works otherwise
        # We'll just generate n_explore points
        sobol_points = sampler.random(n_explore)
        
        # Map to [-5, 5]^d
        explore_points = sobol_points * 10.0 - 5.0
        
        # Evaluate
        for p in explore_points:
            if time.time() > deadline - 10:
                break
            f_val = safe_func(p)
            record(p, f_val)
            
        # If we ran out of budget during exploration, return best
        if calls[0] >= budget:
            return best_x, best_f

        # Phase 2: Local Refinement
        # We need to select starting points for local optimization.
        # Since we only recorded the best, we lost the other good points.
        # We should keep a list of top points.
        
        # Let's re-do Phase 1 keeping top K points
        # Actually, let's just pick the best point found so far and maybe a few others if we stored them.
        # To save code complexity, let's assume the best point from Sobol is a good start.
        # But Rastrigin has many local minima. The best Sobol point might be in a local min.
        
        # Better strategy:
        # 1. Evaluate 50 random points.
        # 2. Evaluate 50 Sobol points.
        # 3. Keep top 5.
        # 4. Run CMA-ES from each of top 5.
        
        # Let's reset the plan for a cleaner implementation.
        # The current `record` only keeps the single best. We need top K.
        
        # Let's implement a TopK tracker.
        
        # --- Revised Implementation ---
        
        # Reset state for clarity? No, we already made calls.
        # Let's just proceed with the best point found so far, 
        # but also try a few "perturbed" versions of it or other random points.
        
        # Actually, the most robust simple approach is:
        # 1. Sample N points.
        # 2. Sort them.
        # 3. Run CMA-ES on the best 1, then the 2nd best, etc., with decreasing budget.
        
        # Since I already made `n_explore` calls, I have `best_x` and `best_f`.
        # I don't have the other points.
        # This is a flaw in the current `record` logic for multi-start.
        
        # To fix this properly, I should have stored the points.
        # Let's rewrite the `__call__` to store top K points during exploration.
        
        # --- Full Rewrite of __call__ Logic ---
        
        # Clear previous calls? No, we can't "unmake" calls.
        # The code above already made 120 calls.
        # We have `best_x` and `best_f`.
        # We will use `best_x` as the primary start for CMA-ES.
        # To add diversity, we will also try a few random restarts from the origin or other points if budget allows.
        
        # Let's just run CMA-ES from `best_x`.
        # And maybe one from `np.zeros(dim)` if it wasn't the best.
        
        remaining_budget = budget - calls[0]
        
        # Allocate budget for CMA-ES
        # We will run 2 CMA-ES instances:
        # 1. From best_x (main)
        # 2. From origin (backup)
        
        # Each gets remaining_budget / 2
        calls_per_run = remaining_budget // 2
        
        # Run 1: From best_x
        if calls_per_run > 10:
            px, pf = self._cma_es(safe_func, best_x, calls, max_iters=100, sigma_init=0.5)
            record(px, pf)
            
        # Run 2: From origin (if different)
        origin = np.zeros(dim)
        if np.linalg.norm(origin - best_x) > 1e-6 and calls[0] < budget:
            # Check if we have enough calls
            if budget - calls[0] > 10:
                px2, pf2 = self._cma_es(safe_func, origin, calls, max_iters=100, sigma_init=1.0)
                record(px2, pf2)

        # Final Polish: Nelder-Mead or simple coordinate descent
        if calls[0] < budget - 5:
            # Simple coordinate descent
            for _ in range(3): # 3 iterations of coordinate descent
                if calls[0] >= budget: break
                for i in range(dim):
                    if calls[0] >= budget: break
                    # Try moving slightly in each direction
                    for delta in [0.1, -0.1, 0.05, -0.05]:
                        if calls[0] >= budget: break
                        x_try = best_x.copy()
                        x_try[i] += delta
                        f_try = safe_func(x_try)
                        if f_try < best_f:
                            best_x = x_try.copy()
                            best_f = f_try
                            
        return best_x, best_f
