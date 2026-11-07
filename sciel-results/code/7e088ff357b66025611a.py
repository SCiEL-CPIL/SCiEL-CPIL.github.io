"""Improved optimizer for BBOB f19 (Griewank-Rosenbrock) using a portfolio approach."""
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES style optimizer with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0

        # Track best
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = func(x)
            calls_made += 1
            return x, float(fx)

        def cma_es(x0, sigma0, n_evals, rng):
            """Simple CMA-ES implementation."""
            nonlocal best_x, best_f, calls_made, deadline

            dim = self.dim
            x = np.array(x0, dtype=np.float64)
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            cs = np.zeros(dim)

            # Learning rates
            mu = dim // 2 + 1
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            csigma = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cc) / (dim + 5)
            damps = 1 + 2 * max(0, np.sqrt(mu_eff / (dim + 1)) - 1) + cc

            # Cholesky of C
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                B = np.eye(dim)

            best_x_local = x.copy()
            best_f_local = np.inf

            for i in range(n_evals):
                if calls_made + mu > budget:
                    break
                if time.time() > deadline:
                    break

                # Generate samples
                ys = []
                for k in range(mu):
                    z = rng.standard_normal(dim)
                    y = x + sigma * (B @ z)
                    y = np.clip(y, -5, 5)
                    ys.append(y)

                # Evaluate
                fvals = []
                for y in ys:
                    _, fy = safe_call(y)
                    fvals.append(fy)
                    if fy < best_f_local:
                        best_f_local = fy
                        best_x_local = y.copy()

                # Sort by fitness
                order = np.argsort(fvals)
                ys = [ys[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Update mean
                new_mean = sum(w[k] * ys[k] for k in range(mu))
                
                # Update path
                pc = (1 - cc) * pc + cc * (new_mean - x) / (sigma * np.linalg.norm(B))
                cs = (1 - csigma) * cs + np.sign(cc * (1 - cc)) * np.linalg.norm(pc) / np.linalg.norm(np.eye(dim))
                
                # Update step size
                sigma *= np.exp(csigma / damps * (np.linalg.norm(pc) / np.sqrt(1 - (1 - cc) ** (2 * (i + 1))) - 1))
                
                # Update covariance
                delta_s = (1 + 2 * max(0, cs - 1)) * csigma
                outer = np.outer(pc, pc)
                C = (1 + c1 * delta_s - c1 - cmu) * C
                C += c1 * (outer - C)
                for k in range(mu):
                    yk = (ys[k] - x) / sigma
                    C += cmu * w[k] * np.outer(yk, yk)

                x = new_mean

                # Update Cholesky
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    B = np.eye(dim)

                # Update global best
                if best_f_local < best_f:
                    best_f = best_f_local
                    best_x = best_x_local.copy()

            return best_x_local, best_f_local

        # Phase 1: Random restarts to find a good starting point
        n_restarts = min(5, budget // 50)
        rng = np.random.default_rng(42)
        
        for r in range(n_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            
            x0 = rng.uniform(-4, 4, dim)
            x0, f0 = safe_call(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Phase 2: CMA-ES from best point
        remaining = budget - calls_made
        if remaining > 20 and best_x is not None:
            sigma0 = 1.0
            cma_es(best_x, sigma0, remaining, rng)

        # Phase 3: Local refinement with (1+1)-ES from best
        remaining = budget - calls_made
        if remaining > 10 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            rng2 = np.random.default_rng(123)
            
            for _ in range(remaining):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng2.standard_normal(dim), -5, 5)
                _, fy = safe_call(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8

        return best_x, best_f
