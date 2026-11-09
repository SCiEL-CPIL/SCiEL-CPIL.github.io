import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        budget = self.budget
        dim = self.dim

        calls_made = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        all_points = []

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError()
            if calls_made >= budget:
                raise ValueError("Budget exceeded")
            calls_made += 1
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            all_points.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def halton_sequence(n, dim):
            """Generate n Halton sequence points in [0,1)^dim."""
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            points = []
            for i in range(1, n + 1):
                point = []
                for d in range(dim):
                    prime = primes[d % len(primes)]
                    f = 1.0
                    ip = i
                    while ip > 0:
                        f /= prime
                        ip = int(ip / prime)
                    point.append(f)
                points.append(point)
            return np.array(points)

        # Phase 1: Deterministic Halton sequence exploration
        n_halton = 45
        halton_pts = halton_sequence(n_halton, dim)
        # Map from [0,1)^dim to [-5,5]^dim
        halton_pts_mapped = -5.0 + 10.0 * halton_pts

        for i in range(n_halton):
            if time.time() > deadline - 20 or calls_made >= budget - 50:
                break
            eval_func(halton_pts_mapped[i])

        # Sort by fitness
        all_points.sort(key=lambda p: p[0])

        # Get top diverse starting points
        start_points = []
        min_dist = 2.0
        for f, x in all_points:
            if len(start_points) >= 5:
                break
            if all(np.linalg.norm(x - sp) > min_dist for sp in start_points):
                start_points.append(x)

        # Fill if needed
        while len(start_points) < 5 and len(all_points) > len(start_points):
            f, x = all_points[len(start_points)]
            if all(np.linalg.norm(x - sp) > 0.5 for sp in start_points):
                start_points.append(x)

        # Phase 2: CMA-ES refinement
        def cma_es(eval_func, x0, budget_limit, dim, deadline):
            import numpy as np
            
            # Parameters
            lam = 4 + int(3 * np.log(dim))
            if lam > budget_limit // 2:
                lam = max(2, budget_limit // 2)
                
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c = (mu_eff + 2) / ((dim + 1)**2 + mu_eff)
            d = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c
            
            cs = (mu_eff + 1.5) / (dim + mu_eff + 5)
            ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
            
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
            
            m = np.array(x0, dtype=float)
            sigma = 1.0 # Initial step size
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            B, D = np.linalg.eigh(C)
            D = np.maximum(D, 1e-12)
            inv_sqrt_C = (B / np.sqrt(D)) @ B.T
            
            local_calls = 0
            best_f = float('inf')
            best_x = m.copy()
            
            while local_calls < budget_limit:
                if time.time() > deadline - 5:
                    break
                    
                ys = []
                xs = []
                fs = []
                
                for i in range(lam):
                    if local_calls >= budget_limit or time.time() > deadline - 5:
                        break
                    
                    z = np.random.randn(dim)
                    y = B @ (D * z)
                    x = m + sigma * y
                    x = np.clip(x, -5, 5)
                    
                    fx = float(eval_func(x))
                    local_calls += 1
                    ys.append(y)
                    xs.append(x)
                    fs.append(fx)
                    
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        
                if len(xs) == 0:
                    break
                    
                ys = np.array(ys)
                xs = np.array(xs)
                fs = np.array(fs)
                
                # Sort by fitness
                idx = np.argsort(fs)
                ys = ys[idx]
                xs = xs[idx]
                fs = fs[idx]
                
                # Update mean
                m_old = m.copy()
                m = sum(weights[i] * xs[i] for i in range(mu))
                
                # Update sigma (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * inv_sqrt_C @ (m - m_old) / sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update C (CMA)
                hc = (1 - np.exp(-2 * (mu_eff - 1) / dim)) * (np.dot(ps, ps) / dim - 1) < (1 + 2 * (mu_eff - 1) / dim)
                pc = (1 - c) * pc + hc * np.sqrt(c * (2 - c) * mu_eff) * (m - m_old) / sigma
                
                c_cma = (mu_eff + 2) / ((dim + 1)**2 + mu_eff)
                nu = 2 * (dim + 1) - mu_eff
                
                y_w = (m - m_old) / sigma
                
                C_update = (1 + nu) * np.outer(y_w, y_w)
                for i in range(mu):
                    y_i = ys[i]
                    C_update += weights[i] * np.outer(y_i - y_w, y_i - y_w)
                    
                C = (1 - c_cma) * C + c_cma * C_update
                
                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                
                # Update B and D
                B, D = np.linalg.eigh(C)
                D = np.maximum(D, 1e-12)
                inv_sqrt_C = (B / np.sqrt(D)) @ B.T

            return best_x, best_f, local_calls

        remaining_budget = budget - calls_made
        if remaining_budget > 10 and time.time() < deadline - 10:
            try:
                best_x, best_f, _ = cma_es(eval_func, best_x, remaining_budget, dim, deadline)
            except Exception:
                pass

        return best_x, best_f
