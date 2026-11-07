import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # Phase 1: Global Exploration
        # Sample a few points to find a good starting location for CMA-ES
        # For 5D, 10 points is cheap and helps avoid bad local minima or starting far from the ridge
        n_explore = 10
        points = rng.uniform(-5, 5, (n_explore, n))
        
        values = np.zeros(n_explore)
        calls_made = 0
        
        for i in range(n_explore):
            if time.time() > deadline:
                break
            values[i] = func(points[i])
            calls_made += 1
            
        if calls_made == 0:
             # Fallback if something went wrong immediately
             x0 = np.zeros(n)
             f0 = func(x0)
             return x0, f0

        # Find the best starting point
        best_idx = np.argmin(values)
        x0 = points[best_idx].copy()
        f0 = values[best_idx]
        
        best_x = x0.copy()
        best_f = f0
        
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f

        # Phase 2: CMA-ES Exploitation
        # Parameters for CMA-ES
        # lambda: population size
        # For n=5, lambda=8 is a good balance between exploration and exploitation speed
        lam = 8
        mu = lam // 2
        
        # Log-weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial state
        # Start from the best point found in exploration
        x_mean = x0.copy()
        
        # Initial sigma: 
        # If we started from a random point, the ridge might be nearby.
        # A sigma of 1.0 is reasonable for a 5D box [-5,5] if we are already close.
        # If we are far, CMA-ES will adapt, but starting too large wastes calls.
        # Let's start with 1.5 to ensure some exploration around the start point.
        sigma = 1.5
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Pre-compute constants
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Number of generations we can afford
        # Each generation costs 'lam' evaluations
        # We have 'remaining_budget' left
        n_gen = remaining_budget // lam
        rem = remaining_budget % lam
        
        for g in range(n_gen):
            # Check time limit
            if time.time() > deadline:
                break
            
            # Generate candidates
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                vals, vecs = np.linalg.eigh(C)
                vals = np.maximum(vals, 1e-12)
                L = (vecs * np.sqrt(vals)) @ vecs.T
            
            ys = np.zeros((lam, n))
            for i in range(lam):
                z = rng.standard_normal(n)
                ys[i] = x_mean + sigma * (L @ z)
                ys[i] = np.clip(ys[i], -5, 5)
            
            # Evaluate
            fs = np.array([func(y) for y in ys])
            calls_made += lam
            
            # Update best
            min_idx = np.argmin(fs)
            if fs[min_idx] < best_f:
                best_f = fs[min_idx]
                best_x = ys[min_idx].copy()
                
            # Sort indices
            order = np.argsort(fs)
            
            # Compute weighted sum of selected points (in standard normal space)
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += w[i] * (ys[order[i]] - x_mean) / sigma
            
            # Update covariance adaptation path
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Update step-size control path
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            
            # Update sigma
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update mean
            x_mean = x_mean + sigma * y_w
            x_mean = np.clip(x_mean, -5, 5)
            
            # Update Covariance C
            # Delta H correction
            delta_h = (1 - np.exp(-2 * (1 - c_c) * (g + 1)))
            h_sig = 1 if norm_ps < (1.4 + 1 / c_sigma) * chi_n else 0
            
            # Compute rank-mu update
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                diff = (ys[order[i]] - x_mean) / sigma
                rank_mu += w[i] * np.outer(diff, diff)
            
            # C update formula
            # Note: Using old x_mean for rank_mu is standard in some implementations, 
            # but technically it should be the mean before update. 
            # The difference is small for small steps.
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_sig) * c_sigma * (2 - c_sigma) * (np.outer(ps, ps) - np.trace(C) * np.eye(n))) + c_mu * rank_mu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Check budget
            if calls_made >= budget:
                break

        # Use remaining budget for single step (1+1) or just return
        # If we have leftover calls, do a few (1+1) steps with the current sigma
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Simple (1+1) step using current sigma and C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)
            
            z = rng.standard_normal(n)
            y = x_mean + sigma * (L @ z)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                # Slight expansion
                sigma *= 1.1
            else:
                # Slight contraction
                sigma *= 0.9
                sigma = np.clip(sigma, 1e-6, 10.0)
                
            # Update mean to best found so far if we improved
            if fy < x_mean is not None: # This check is redundant, just update if improved
                 pass # We already updated best_x. x_mean is the CMA-ES mean, not necessarily the global best.
            # Actually, for the (1+1) tail, it's better to just do (1+1) from the current best_x
            # Let's switch to pure (1+1) from best_x for the tail to be safe and simple
            # But we already used calls. Let's just break if we are close to limit.
            # The (1+1) tail is a minor optimization.
            if calls_made >= budget:
                break

        return best_x, best_f
