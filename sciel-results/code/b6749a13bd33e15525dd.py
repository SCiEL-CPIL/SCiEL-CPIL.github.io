"""CMA-ES with adaptive restarts and local polish for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with adaptive restarts and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget:
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                return float(func(x_clipped)), x_clipped
            except Exception:
                return None, None

        def update(x, f):
            nonlocal best_x, best_f
            if f is not None and f < best_f:
                best_f = f
                best_x = x.copy()

        def cma_es(x0, sigma0, max_calls, rng):
            """Run CMA-ES with call tracking. Returns (best_x, best_f, calls_used)."""
            n = dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            sigma = sigma0

            x = x0.copy()
            calls_used = 0
            local_best_x = x.copy()
            local_best_f = np.inf

            # Initial population
            pop = np.zeros((lam, n))
            fvals = np.zeros(lam)
            for i in range(lam):
                if check_time() or calls_used >= max_calls:
                    break
                z = rng.standard_normal(n)
                x_trial = x + sigma * z
                res = call_func(x_trial)
                if res[0] is None:
                    return local_best_x, local_best_f, calls_used
                f, x_eval = res
                calls_used += 1
                pop[i] = x_eval
                fvals[i] = f
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = x_eval.copy()

            # Main loop
            while calls_used < max_calls and not check_time():
                # Sort by fitness
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                # Update mean
                x_new = np.zeros(n)
                for i in range(mu):
                    x_new += weights[i] * pop[i]
                x = x_new

                # Update ps and pc
                # Need Cinv_sqrt for ps update. Approximate with eigendecomposition.
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    Cinv_sqrt = np.eye(n)

                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * Cinv_sqrt @ (x - x_new)
                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (calls_used // lam)))) < (1.4 + 2.0 / (n + 1)) * chi_n
                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * (x - x_new)

                # Update covariance
                # Rank-1 update
                C = (1.0 + c1 * hsig * (1.0 - hsig) - c1 - cmu) * C + c1 * np.outer(pc, pc)
                # Rank-mu update
                for i in range(mu):
                    y_i = (pop[i] - x) / sigma
                    C += cmu * weights[i] * np.outer(y_i, y_i)

                # Ensure symmetry
                C = (C + C.T) / 2.0

                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-15, 1e6)

                # Generate new population
                for i in range(lam):
                    if calls_used >= max_calls or check_time():
                        break
                    z = rng.standard_normal(n)
                    x_trial = x + sigma * (Cinv_sqrt @ z)
                    res = call_func(x_trial)
                    if res[0] is None:
                        return local_best_x, local_best_f, calls_used
                    f, x_eval = res
                    calls_used += 1
                    pop[i] = x_eval
                    fvals[i] = f
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x_eval.copy()

                # Convergence check
                if sigma < 1e-10:
                    break

            return local_best_x, local_best_f, calls_used

        def polish(x0, max_calls):
            """Local polish using Nelder-Mead with call tracking."""
            nonlocal n_calls, best_x, best_f
            if check_time() or n_calls >= budget:
                return x0.copy()
            
            # We need to track calls inside minimize
            # scipy's minimize doesn't easily allow us to intercept calls without wrapping
            # We'll wrap the function to count calls
            
            local_calls = 0
            def wrapped_func(x):
                nonlocal local_calls
                if check_time() or (n_calls + local_calls) >= budget:
                    raise ValueError("Budget or time exceeded")
                local_calls += 1
                x_clipped = np.clip(x, -5, 5)
                return float(func(x_clipped))
            
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead',
                              options={'maxiter': 50, 'xatol': 1e-10, 'fatol': 1e-10})
                # Update global call count
                n_calls += local_calls
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
                return np.clip(res.x, -5, 5).copy()
            except Exception:
                # If polish fails, just return the starting point
                return x0.copy()

        rng = np.random.default_rng(42)

        # Phase 1: Initial random sampling to find promising basins
        n_initial = min(10, budget // 50)
        for i in range(n_initial):
            if check_time() or n_calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            res = call_func(x)
            if res[0] is None:
                break
            f, x_eval = res
            update(x_eval, f)

        if best_x is None:
            x = np.zeros(dim)
            res = call_func(x)
            if res[0] is not None:
                update(res[1], res[0])
            if best_x is None:
                return None

        # Phase 2: CMA-ES restarts
        remaining = budget - n_calls
        if remaining < 10:
            return best_x, best_f

        # Strategy: Run CMA-ES from best point with large budget,
        # then if budget remains, try other starts
        n_restarts = 3
        calls_per_restart = remaining // n_restarts
        calls_per_restart = max(calls_per_restart, 10)

        starts = [best_x.copy()]
        # Add a few random starts
        for _ in range(n_restarts - 1):
            starts.append(rng.uniform(-5, 5, dim))

        for i, x0 in enumerate(starts):
            if check_time() or n_calls >= budget:
                break
            # First restart gets more budget
            if i == 0:
                max_calls = min(remaining - (n_restarts - 1) * calls_per_restart, budget - n_calls)
            else:
                max_calls = min(calls_per_restart, budget - n_calls)

            if max_calls < 10:
                break

            # Initial sigma: larger for first restart, smaller for later
            sigma0 = 2.0 if i == 0 else 1.0

            res_x, res_f, used = cma_es(x0, sigma0, max_calls, rng)
            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()

        # Phase 3: Local polish with remaining budget
        if not check_time() and n_calls < budget:
            polish_calls = min(50, budget - n_calls)
            if polish_calls > 0:
                polished_x = polish(best_x.copy(), polish_calls)
                # best_x and best_f are updated inside polish if improved

        return best_x, best_f
# EVOLVE-BLOCK-END
