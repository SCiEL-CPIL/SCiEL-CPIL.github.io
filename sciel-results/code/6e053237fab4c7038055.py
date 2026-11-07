"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES with restarts for BBOB f21 (Gallagher 101 peaks).
    
    Strategy:
    1. Global exploration with diverse starting points.
    2. CMA-ES with small population (λ=10) and aggressive restarts
       to model correlated structure and adaptively navigate multimodal landscape.
    3. Adaptive step size and covariance matrix to exploit local structure.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        calls_made = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            try:
                val = float(func(x))
            except Exception:
                return None
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Global exploration - diverse starting points
        n_starts = max(10, min(40, budget // 5))
        starts = []
        for i in range(n_starts):
            if not check_time() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is None:
                break
            starts.append((fx, x.copy()))

        if best_x is None:
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is not None:
                best_f = fx
                best_x = x.copy()
                starts = [(fx, x.copy())]

        # Sort starting points by function value (best first)
        starts.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES with restarts for each promising start
        # Use remaining budget wisely
        remaining = budget - calls_made
        n_local = min(len(starts[:8]), max(1, remaining // 30))
        
        for i in range(n_local):
            if not check_time() or calls_made >= budget:
                break
            f0, x0 = starts[i]
            
            # Run CMA-ES with restarts
            cma_result = self._run_cma_with_restarts(func, x0, f0, remaining - calls_made, dim, rng, check_time, safe_call, calls_made)
            if cma_result is not None:
                calls_made = cma_result[0]

        # Phase 3: Final refinement with (1+1)-ES around best point
        if best_x is not None and check_time() and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            no_improve = 0
            max_no_improve = 80
            
            while check_time() and calls_made < budget:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma *= 1.03
                else:
                    sigma *= 0.97
                    no_improve += 1
                
                sigma = max(sigma, 1e-6)
                sigma = min(sigma, 2.0)
                
                if no_improve > max_no_improve:
                    break

        return best_x, best_f

    def _run_cma_with_restarts(self, func, x0, f0, budget_left, dim, rng, check_time, safe_call, calls_made):
        """
        Run CMA-ES with aggressive restarts.
        Returns updated calls_made count.
        """
        lambda_ = 10  # Population size
        mu = lambda_ // 2  # Number of parents
        max_evals_per_run = 50  # Restart every 50 evaluations if no improvement
        
        # CMA-ES parameters
        sigma0 = 1.0
        alpha_mu = 1.0 + 2.0 / (dim + 2.0)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        mu_eff = mu / (2.0 ** (1.0 / mu))
        
        # Initialize CMA-ES state
        x_mean = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        inv_sqrt_C = np.eye(dim)
        
        # Track best
        local_best_f = f0
        local_best_x = x0.copy()
        no_improve = 0
        
        for gen in range(max(1, budget_left // lambda_)):
            if not check_time() or calls_made >= self.budget:
                break
            
            # Generate population
            w = np.zeros(mu)
            w[:mu] = np.log(mu) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            
            # Sample lambda points
            ys = []
            xs = []
            fs = []
            
            for i in range(lambda_):
                if not check_time() or calls_made >= self.budget:
                    break
                
                # Sample from N(0, C)
                z = rng.standard_normal(dim)
                y = inv_sqrt_C @ z
                x = x_mean + sigma * y
                x = np.clip(x, -5, 5)
                
                fx = safe_call(x)
                if fx is None:
                    break
                
                ys.append(y)
                xs.append(x)
                fs.append(fx)
            
            if len(fs) < lambda_:
                break
            
            # Sort by function value
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            xs = [xs[i] for i in order]
            fs = [fs[i] for i in order]
            
            # Update best
            if fs[0] < local_best_f:
                local_best_f = fs[0]
                local_best_x = xs[0].copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Check for restart
            if no_improve >= max_evals_per_run:
                break
            
            # Update mean
            x_mean_new = sum(w[i] * xs[i] for i in range(mu))
            
            # Update step size
            y_mean = sum(w[i] * ys[i] for i in range(mu))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * y_mean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            delta_h = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_c * (2 - c_c - mu_eff) + delta_h) * C
            C += c_c * (np.outer(pc, pc) - C)
            
            # Update inverse square root
            # Simple approach: recompute from eigen decomposition every few iterations
            if gen % 10 == 0:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except:
                    inv_sqrt_C = np.eye(dim)
            
            x_mean = x_mean_new
        
        return calls_made
# EVOLVE-BLOCK-END
