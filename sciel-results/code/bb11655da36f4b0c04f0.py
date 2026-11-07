"""Combined optimizer: Sobol quasi-random mapping + CMA-ES refinement + Powell polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Sobol quasi-random mapping + CMA-ES refinement + Powell polish for BBOB f19.
    
    Uses dense Sobol sequence for systematic space coverage to locate the narrow
    Rosenbrock valley amidst Griewank oscillations, then refines with CMA-ES
    and polishes with Powell's method.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _sobol_scan(self, func, budget, deadline):
        """Dense Sobol quasi-random scan to systematically cover the search space."""
        dim = self.dim
        calls_used = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        # Generate Sobol sequence - use as many points as budget allows
        # Sobol sequences provide low-discrepancy coverage
        n_points = min(budget, 256)  # Power of 2 for good Sobol properties
        
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        samples = sampler.random(n_points)
        
        # Map from [0,1]^dim to [-5,5]^dim
        samples = samples * 10.0 - 5.0
        
        # Evaluate all points
        for i in range(n_points):
            if calls_used >= budget or time.time() >= deadline:
                break
            x = samples[i]
            f = func(x)
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        return best_x, best_f, calls_used

    def _cmaes_run(self, func, x0, sigma0, budget, deadline):
        """Run CMA-ES from x0 with sigma0, using up to budget calls."""
        dim = self.dim
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        calls_used = 0
        best_x = x.copy()
        best_f = func(x)
        calls_used += 1

        for gen in range(500):
            if calls_used >= budget or time.time() >= deadline:
                break

            # Generate population using Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            z = np.random.default_rng(gen * 1000).standard_normal((lam, dim))
            pop = np.array([x + sigma * (zi @ L) for zi in z])
            pop = np.clip(pop, -5, 5)
            fits = np.array([func(p) for p in pop])
            calls_used += lam

            # Update best
            idx_min = np.argmin(fits)
            if fits[idx_min] < best_f:
                best_f = fits[idx_min]
                best_x = pop[idx_min].copy()

            # Sort by fitness
            sorted_idx = np.argsort(fits)
            y_sorted = pop[sorted_idx]

            # Weighted mean
            m_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

            # Update path states
            yw = (m_new - x) / sigma
            pc = (1 - cs) * pc + cs * yw
            ps = (1 - cs) * ps + cs * np.sign(yw)

            # Sigma update
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi
            sigma *= np.exp(cs / chi * (np.linalg.norm(pc) / chi - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)

            # Covariance update
            delta_hsig = (1.0 if hsig else 0.0) * (1 - cs) * cs
            C = C + c1 * (np.outer(yw, yw) + (1.0 if hsig else 0.0) * cs * (1 - cs) - C)
            C += cmu * np.sum(weights[:, None] * np.outer(y_sorted[:mu] - x, y_sorted[:mu] - x) / (sigma ** 2), axis=0)
            C = (C + C.T) / 2

            # Ensure positive definite
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += (1e-12 - np.min(eigvals) + 1e-12) * np.eye(dim)

            x = m_new

            # Early stop if sigma is very small
            if sigma < 1e-8:
                break

        return best_x, best_f, calls_used

    def _powell_polish(self, func, x0, budget, deadline):
        """Powell's method polish with call counting."""
        if budget < 10 or time.time() >= deadline:
            return x0, func(x0), 1
        
        calls_used = 0
        best_x = x0.copy()
        best_f = func(x0)
        calls_used += 1
        
        count = [calls_used]
        def counted_func(x):
            nonlocal best_x, best_f
            count[0] += 1
            if time.time() >= deadline or count[0] >= budget:
                raise StopIteration
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        try:
            res = minimize(counted_func, best_x, method='Powell',
                          options={'maxiter': 100, 'xtol': 1e-12, 'ftol': 1e-12})
        except StopIteration:
            pass
        except Exception:
            pass
        
        calls_used = count[0]
        return best_x, best_f, calls_used

    def __call__(self, func):
        deadline = time.time() + 320.0
        dim = self.dim
        budget = self.budget

        global_best_x = np.zeros(dim)
        global_best_f = np.inf
        calls_remaining = budget

        # Phase 1: Sobol quasi-random scan for systematic space coverage
        # Use ~40% of budget for initial mapping
        sobol_budget = int(budget * 0.4)
        if sobol_budget < 20:
            sobol_budget = budget
        
        sx, sf, sused = self._sobol_scan(func, sobol_budget, deadline)
        calls_remaining -= sused
        global_best_x = sx.copy()
        global_best_f = sf

        # Phase 2: CMA-ES refinement from the best Sobol point
        # Use ~40% of remaining budget
        if calls_remaining > 10 and time.time() < deadline:
            cma_budget = int(calls_remaining * 0.4)
            if cma_budget < 15:
                cma_budget = calls_remaining
            
            # Try CMA-ES with a few different sigma values
            sigma_configs = [2.0, 1.0, 0.5]
            for sigma0 in sigma_configs:
                if calls_remaining <= 5 or time.time() >= deadline:
                    break
                
                alloc = min(cma_budget, calls_remaining)
                cx, cf, cused = self._cmaes_run(func, global_best_x, sigma0, alloc, deadline)
                calls_remaining -= cused
                
                if cf < global_best_f:
                    global_best_f = cf
                    global_best_x = cx.copy()
                
                # Stop early if we've found a very good solution
                if global_best_f < 1e-6:
                    break

        # Phase 3: Powell polish with remaining budget
        if calls_remaining > 10 and time.time() < deadline:
            px, pf, pused = self._powell_polish(func, global_best_x, calls_remaining, deadline)
            if pf < global_best_f:
                global_best_f = pf
                global_best_x = px.copy()

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
