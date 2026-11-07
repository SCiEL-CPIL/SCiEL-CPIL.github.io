"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer with multi-start and local polish for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 320.0  # Safety margin before 330s kill

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))  # lam = 8 for dim=5
        mu = lam // 2  # mu = 4

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)

        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = (2 * (mueff - 2 + 2 / mueff)) / ((dim + 2)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))

        def cmaes_local(x0, sigma0, max_evals, rng, evals_used):
            """Run CMA-ES from x0 with sigma0, using at most max_evals evaluations."""
            m = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            best_x = m.copy()
            best_f = func(best_x)
            evals_used += 1
            
            if max_evals <= 1:
                return best_x, best_f, evals_used

            while evals_used < max_evals:
                if time.time() - start_time > time_limit:
                    break
                
                # Sample population
                ys = rng.standard_normal((lam, dim))
                
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                xs = m[None, :] + sigma * (ys @ L.T)
                xs = np.clip(xs, -5, 5)
                
                # Evaluate
                fs = np.array([func(x) for x in xs])
                evals_used += lam
                
                idx_best = np.argmin(fs)
                if fs[idx_best] < best_f:
                    best_f = fs[idx_best]
                    best_x = xs[idx_best].copy()
                
                # Sort
                idx_sorted = np.argsort(fs)
                x_sorted = xs[idx_sorted]
                
                # Weighted mean
                new_m = np.dot(weights, x_sorted[:mu])
                
                # Update step size
                diff = (new_m - m) / sigma
                v = np.linalg.solve(L.T, diff)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * v
                
                hsig = 1.0 if np.linalg.norm(ps) < (1.4 + 2.0 / (dim + 1.0)) * chiN else 0.0
                sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1.0))
                sigma = np.clip(sigma, 1e-12, 10.0)
                
                # Update covariance
                pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * diff
                rank1 = np.outer(pc, pc)
                v_mu = (x_sorted[:mu] - m) / sigma
                rankmu = np.dot((weights[:, None] * v_mu).T, v_mu)
                
                C = (1 + c1 * hsig * (2 - hsig) - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
                C = (C + C.T) / 2.0
                m = new_m
                
                # Check convergence
                if sigma < 1e-10:
                    break
            
            return best_x, best_f, evals_used

        def local_polish(x0, max_evals, rng, evals_used):
            """Simple coordinate descent / random walk polish."""
            best_x = x0.copy()
            best_f = func(best_x)
            evals_used += 1
            
            if max_evals <= 1:
                return best_x, best_f, evals_used
            
            step = 0.1
            for _ in range(max_evals - 1):
                if time.time() - start_time > time_limit:
                    break
                if evals_used >= max_evals:
                    break
                
                # Try coordinate-wise moves
                improved = False
                for d in range(dim):
                    for sign in [1, -1]:
                        x_try = best_x.copy()
                        x_try[d] += sign * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        evals_used += 1
                        if f_try < best_f:
                            best_x = x_try
                            best_f = f_try
                            improved = True
                        if evals_used >= max_evals:
                            break
                    if evals_used >= max_evals:
                        break
                
                if not improved:
                    step *= 0.5
                    if step < 1e-8:
                        break
                else:
                    step = max(step * 1.2, 0.01)
            
            return best_x, best_f, evals_used

        # Multi-start CMA-ES
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        starts = [np.zeros(dim)]
        for i in range(5):
            starts.append(rng.uniform(-4, 4, size=dim))
        
        # Allocate budget: give more to the first (center) start
        # 500 total: center gets ~200, others get ~60 each
        n_starts = len(starts)
        base_budget = 60
        center_budget = 200
        
        best_x_global = None
        best_f_global = np.inf
        total_evals = 0
        
        for i, x0 in enumerate(starts):
            if time.time() - start_time > time_limit - 10:
                break
            
            if total_evals >= budget:
                break
            
            if i == 0:
                alloc = min(center_budget, budget - total_evals)
            else:
                alloc = min(base_budget, budget - total_evals)
            
            if alloc < 2:
                break
            
            sigma0 = 2.0 if i == 0 else 1.5
            bx, bf, evals = cmaes_local(x0, sigma0, alloc, rng, total_evals)
            total_evals = evals
            
            if bf < best_f_global:
                best_f_global = bf
                best_x_global = bx.copy()
        
        # Local polish with remaining budget
        remaining = budget - total_evals
        if remaining > 0 and best_x_global is not None:
            if time.time() - start_time < time_limit - 5:
                bx, bf, evals = local_polish(best_x_global, remaining, rng, total_evals)
                total_evals = evals
                if bf < best_f_global:
                    best_f_global = bf
                    best_x_global = bx.copy()
        
        return best_x_global, best_f_global

# EVOLVE-BLOCK-END
