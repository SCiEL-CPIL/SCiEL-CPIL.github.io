"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """LHS multi-start followed by CMA-ES refinement with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit, use 290s to be safe
        deadline = start_time + 290.0 
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            val = func(x)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)

        # Phase 1: LHS global sampling for diverse starting points
        # Use 25% of budget for exploration, min 20, max 50
        n_starts = max(20, min(50, budget // 4))
        
        for i in range(n_starts):
            if not check_time() or calls_made >= budget:
                break
            # Latin Hypercube Sampling for dim=5
            x = np.empty(dim)
            for d in range(dim):
                perm = rng.permutation(n_starts)
                x[d] = -5.0 + 10.0 * (perm[i] + rng.random()) / n_starts
            x = np.clip(x, -5, 5)
            fx = safe_call(x)
            if fx is None:
                break

        if best_x is None:
            # Fallback: single random start
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is not None:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES refinement from best point
        if best_x is not None and check_time() and calls_made < budget:
            x_mean = best_x.copy()
            f_best = best_f
            
            # CMA-ES parameters
            # Population size: lambda = 4 + 3 ln(n)
            lam = max(4, int(4 + 3 * np.log(dim)))
            # Ensure we don't exceed remaining budget in one generation
            # If remaining budget < lam, we might do a smaller run or stop
            if budget - calls_made < lam:
                 lam = max(1, budget - calls_made)
            
            C = np.eye(dim)
            sigma = 1.0 # Initial step size
            pc = np.zeros(dim)
            cs = 1.0 / (lam + 3)
            c1 = 2 / ((dim + 1) ** 2 + 10)
            cmu = min(1 - c1, 2 * ((dim + 1.7) ** 2 - 1) / ((dim + 2) ** 2 + 13))
            
            # Pre-compute weights
            weights = np.log(np.arange(1, lam + 1))
            weights /= weights.sum()
            mu_eff = 1 / (weights ** 2).sum()
            
            # Evolution path parameters
            pc_n = np.sqrt((1 + 2 * max(0, (1 + cs) / (1 - cs)) - 1) / (1 - cs))
            cs_n = cs * (1 + 2 * max(0, (1 + cs) / (1 - cs)))
            c_sigma = 2 * (mu_eff - 1) / (dim + 1)
            c_mean = 1 + 2 * max(0, np.sqrt(mu_eff / (dim + 1)) - 1)

            while check_time() and calls_made + lam <= budget:
                # Generate population
                ys = []
                zs = []
                for j in range(lam):
                    if not check_time() or calls_made >= budget:
                        break
                    z = rng.standard_normal(dim)
                    # Use Cholesky decomposition for efficient sampling from C
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        # Fallback to eigen decomposition if not positive definite
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-10)
                        L = eigvecs @ np.diag(np.sqrt(eigvals))
                    
                    y = x_mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = safe_call(y)
                    if fy is None:
                        break
                    ys.append((fy, y, z))
                    zs.append(z)

                if len(ys) < lam:
                    break

                # Sort by fitness
                ys.sort(key=lambda t: t[0])
                
                # Update global best
                if ys[0][0] < f_best:
                    f_best = ys[0][0]
                    best_x = ys[0][1].copy()

                # Compute weighted mean
                y_mean = np.zeros(dim)
                z_mean = np.zeros(dim)
                for w, (fy, y, z) in zip(weights, ys):
                    y_mean += w * y
                    z_mean += w * z

                # Update evolution path pc
                pc = (1 - cs) * pc + cs * (z_mean / sigma)
                pc_norm = np.linalg.norm(pc)
                if pc_norm > 1.4 * (1 + cs) * lam:
                    pc = pc * (1.4 * (1 + cs) * lam / pc_norm)

                # Update covariance matrix C
                # Delta sigma term
                delta_sigma = (pc_norm / np.sqrt(2 * dim) - (1 - cs)) / (1 - cs)
                C = (1 + c1 * delta_sigma * dim) * C
                C += c1 * (np.outer(pc, pc) - C)
                
                # Rank-mu update
                for w, (fy, y, z) in zip(weights, ys):
                    C += cmu * (np.outer(z, z) - C)

                # Ensure C is positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update mean
                x_mean = x_mean + c_mean * sigma * (y_mean - x_mean)
                x_mean = np.clip(x_mean, -5, 5)

                # Update sigma
                sigma *= np.exp(cs_n * (pc_norm / np.sqrt(2 * dim) - 1))
                sigma = max(sigma, 1e-6)
                sigma = min(sigma, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
