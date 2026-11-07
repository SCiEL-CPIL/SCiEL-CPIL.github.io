import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, calls, max_iters=80, step0=0.5):
        """Nelder-Mead for final polish."""
        d = self.dim
        x = np.array(x0, dtype=float)
        f0 = func(x)
        simplex = [x.copy()]
        fs = [f0]
        for i in range(d):
            xi = x.copy()
            xi[i] += step0
            simplex.append(xi)
            fs.append(func(xi))
        best_x = x.copy()
        best_f = f0
        
        for _ in range(max_iters):
            if calls[0] >= self.budget:
                break
            order = np.argsort(fs)
            simplex = [simplex[i] for i in order]
            fs = [fs[i] for i in order]
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = simplex[0].copy()
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + (centroid - simplex[-1])
            fr = func(xr)
            if fr < fs[0]:
                # Extension
                xe = centroid + 2.0 * (centroid - simplex[-1])
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fs[-1] = fe
                else:
                    simplex[-1] = xr
                    fs[-1] = fr
                continue
            if fr < fs[-2]:
                simplex[-1] = xr
                fs[-1] = fr
                continue
            
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = func(xc)
            if fc < fs[-1]:
                simplex[-1] = xc
                fs[-1] = fc
                continue
            
            # Shrink
            for i in range(1, d + 1):
                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                fs[i] = func(simplex[i])
                
        return best_x, best_f

    def _cma_es(self, func, x0, calls, budget_limit, sigma_init=1.0, lambda_=10, mu=5):
        """
        Lightweight CMA-ES implementation.
        """
        d = self.dim
        m = np.array(x0, dtype=float)
        sigma = sigma_init
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Weights
        w = np.log(mu / 2 + 1) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        L = np.linalg.cholesky(C)
        
        best_x = m.copy()
        best_f = func(m)
        
        for gen in range(1000):
            if calls[0] >= budget_limit:
                break
            
            # Sample population
            xs = []
            fs = []
            for k in range(lambda_):
                z = np.random.randn(d)
                x = m + sigma * (L @ z)
                xs.append(x)
                fs.append(func(x))
            
            # Sort
            idx = np.argsort(fs)
            xs = [xs[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()
                
            # Update mean
            m_new = sum(w[i] * xs[i] for i in range(mu))
            
            # Update covariance matrix
            v = [(xs[i] - m) / sigma for i in range(mu)]
            C_mu = sum(w[i] * np.outer(v[i], v[i]) for i in range(mu))
            pc_new = (1 - c_c) * pc + c_c * ((m_new - m) / sigma)
            C_1 = (1 - (c_1 + c_mu)) * C + c_1 * (1 + (1 - c_sigma) * d_sigma * np.outer(pc, pc) - c_sigma * np.outer(pc, pc)) + c_mu * (C_mu - C)
            
            # Symmetrize and ensure positive definite
            C = (C_1 + C_1.T) / 2
            C += 1e-10 * np.eye(d)
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ((m_new - m) / sigma)
            
            # Delta sigma
            sigma_new = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * d) - 1))
            
            # Update L
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-3 * np.eye(d)
                L = np.linalg.cholesky(C)
                
            m = m_new
            sigma = sigma_new
            pc = pc_new
            
            if sigma < 1e-6:
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

        # 1. Initial coarse grid search to find promising basins
        # Rastrigin has many local minima. A coarse grid helps identify the region of the global minimum.
        # Grid of 3x3x3x3x3 = 243 points.
        # Range [-4, 4]
        grid_points = []
        for x1 in [-4, 0, 4]:
            for x2 in [-4, 0, 4]:
                for x3 in [-4, 0, 4]:
                    for x4 in [-4, 0, 4]:
                        for x5 in [-4, 0, 4]:
                            grid_points.append(np.array([x1, x2, x3, x4, x5]))
        
        # Shuffle to avoid bias? Or just evaluate.
        # Actually, random sampling might be better for the first few to detect the shift.
        # Let's do a mix: Random 50 points, then Grid.
        
        # Random initial points
        for _ in range(50):
            x = np.random.uniform(-4.5, 4.5, dim)
            f = safe_func(x)
            record(x, f)
            
        # Evaluate Grid
        for x in grid_points:
            if calls[0] >= budget - 100:
                break
            f = safe_func(x)
            record(x, f)
            
        # 2. CMA-ES from the best point found so far
        # We have used ~50 + 243 = 293 calls.
        # Remaining budget: ~207 calls.
        # CMA-ES with lambda=10 uses 10 calls per generation.
        # 207 / 10 = ~20 generations.
        
        if calls[0] < budget - 10:
            # Run CMA-ES
            px, pf = self._cma_es(safe_func, best_x, calls, budget_limit=budget - 20, sigma_init=1.0)
            record(px, pf)
            
        # 3. Final Polish with Nelder-Mead
        if calls[0] < budget - 5:
            px, pf = self._nelder_mead(safe_func, best_x, calls, max_iters=50, step0=0.5)
            record(px, pf)

        return best_x, best_f
