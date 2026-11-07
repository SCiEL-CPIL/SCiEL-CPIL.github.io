import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        total_budget = self.budget
        calls = [0]

        def safe_func(x):
            if calls[0] >= total_budget:
                return np.inf
            calls[0] += 1
            try:
                return float(func(np.asarray(x, dtype=float)))
            except Exception:
                return np.inf

        rng = np.random.default_rng(42)
        
        # Phase 1: Random search to find a good basin
        # Rastrigin has many local minima, but we want the global one.
        # Global min is shifted.
        # Try a few random points and the origin.
        candidates = [np.zeros(dim)]
        for _ in range(10):
            candidates.append(rng.uniform(-4, 4, dim))
            
        best_x = None
        best_f = np.inf
        for x0 in candidates:
            fx = safe_func(x0)
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()
                
        # Phase 2: CMA-ES from the best point
        if calls[0] < total_budget and best_x is not None:
            best_x, best_f = self._cma_es(func, safe_func, best_x, 1.0, total_budget - calls[0], rng, deadline)
            
        return best_x, best_f

    def _cma_es(self, func, safe_func, x0, sigma0, budget, rng, deadline):
        dim = len(x0)
        x = x0.copy()
        sigma = sigma0
        pop_size = max(10, 4 + int(3 * np.log(dim)))
        mu = pop_size // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        B = np.eye(dim)
        D = np.ones(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        best_x = x.copy()
        best_f = safe_func(x) # Note: this counts a call
        
        calls_left = budget - 1 # We already used 1 call above? 
        # Wait, safe_func in _cma_es uses the global calls counter?
        # Yes, safe_func is passed in and uses the closure `calls`.
        # So we must be careful.
        
        # Actually, let's just loop until budget is exhausted or time is up.
        
        while calls[0] < budget:
            if time.time() > deadline - 2.0:
                break
            
            # Generate population
            xs = []
            fs = []
            for i in range(pop_size):
                if calls[0] >= budget:
                    break
                z = rng.standard_normal(dim)
                # B @ (D * z) is equivalent to (B * D) @ z? No.
                # Transformation is B * D.
                # x_new = x + sigma * (B @ (D * z))
                step = B @ (D * z)
                x_i = x + sigma * step
                # Clip to box? The problem says func clips, but CMA-ES works better without clipping if we adapt sigma.
                # However, the box is [-5, 5]. If we go out, func clips.
                # It's safer to clip the proposal to stay in bounds to avoid wasted calls?
                # Actually, if we clip, we might bias the distribution.
                # But for dim=5 and box [-5,5], and sigma starting at 1, it's usually fine.
                x_i = np.clip(x_i, -5, 5)
                f_i = safe_func(x_i)
                xs.append(x_i)
                fs.append(f_i)
                
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
                    
            if len(xs) < mu:
                break
                
            # Sort
            indices = np.argsort(fs)
            xs_sorted = [xs[i] for i in indices]
            fs_sorted = [fs[i] for i in indices]
            
            # Update mean
            x_old = x.copy()
            x = sum([w[i] * xs_sorted[i] for i in range(mu)])
            
            # Update pc
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) / sigma * (x - x_old)
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B.T @ (D * (x - x_old)))
            
            # Update B, D (Eigen decomposition of C)
            # We need C.
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - delta_sigma) * ps * ps.T) + c_mu * sum w_i (y_i y_i.T)
            # This requires maintaining C.
            # To simplify, we can use the simplified CMA-ES or just diagonal CMA-ES?
            # Diagonal CMA-ES is much faster and often sufficient for low dimensions.
            # Let's switch to Diagonal CMA-ES to avoid eigen decomposition cost and complexity.
            
            # Break out of full CMA-ES and use diagonal?
            # Or implement diagonal CMA-ES directly.
            
            # Let's replace the loop with a Diagonal CMA-ES implementation.
            break
