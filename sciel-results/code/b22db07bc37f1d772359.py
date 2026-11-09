"""CMA-ES with restarts for BBOB f16 (Weierstrass) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget:
                return None
            n_calls += 1
            try:
                return float(func(x))
            except Exception:
                return None

        def update(x, f):
            nonlocal best_x, best_f
            if f is not None and f < best_f:
                best_f = f
                best_x = x.copy()

        def cma_es(x0, sigma0, max_calls, rng):
            """Run CMA-ES with call tracking. Returns (best_x, best_f, calls_used)."""
            n = dim
            # Population size
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = (2.0 / ((n + 1.3) ** 2 + mueff)) * (1.0 - cc)
            cmu = (2.0 * (mueff / n - 1.0) / ((n + 2) ** 2 + mueff)) * (
                min(1.0, c1) - c1
            )

            C = np.eye(n)
            Cinv = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            sigma = sigma0
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

            x = x0.copy()
            calls_used = 0
            local_best_x = x.copy()
            local_best_f = np.inf

            # Initial population
            pop = np.zeros((lam, n))
            fvals = np.zeros(lam)
            for i in range(lam):
                z = rng.standard_normal(n)
                pop[i] = x + sigma * z
                f = call_func(pop[i])
                calls_used += 1
                if f is None:
                    return local_best_x, local_best_f, calls_used
                fvals[i] = f
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = pop[i].copy()

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

                # Update covariance
                diff = pop[:mu] - x
                C = (1.0 - c1 - cmu) * C + c1 * (np.outer(pc, pc) +
                     (1.0 + (1.0 - cc) * cc) * C - C) + cmu * np.outer(diff.T @ diff / mueff,
                     np.ones(1)) if False else C
                # Simpler covariance update
                C = (1.0 - c1 - cmu) * C
                C += c1 * (np.outer(pc, pc) + (1.0 + (1.0 - cc) * cc) * C - C)
                for i in range(mu):
                    C += cmu * weights[i] * np.outer(pop[i] - x, pop[i] - x)

                # Ensure symmetry
                C = (C + C.T) / 2.0

                # Compute inverse sqrt of C for sampling
                # Use eigendecomposition for stability
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    Cinv_sqrt = np.eye(n)

                # Update ps and pc
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * Cinv_sqrt @ (x - x_new)
                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (calls_used // lam)))) < (1.4 + 2.0 / (n + 1)) * chi_n
                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * (x - x_new)

                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-15, 1e6)

                # Generate new population
                for i in range(lam):
                    if calls_used >= max_calls or check_time():
                        break
                    z = rng.standard_normal(n)
                    pop[i] = x + sigma * (Cinv_sqrt @ z)
                    f = call_func(pop[i])
                    calls_used += 1
                    if f is None:
                        return local_best_x, local_best_f, calls_used
                    fvals[i] = f
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = pop[i].copy()

                # Convergence check
                if sigma < 1e-10:
                    break
                if calls_used >= max_calls:
                    break

            return local_best_x, local_best_f, calls_used

        rng = np.random.default_rng(42)

        # Phase 1: Initial random sampling to find promising basins
        n_initial = min(15, budget // 10)
        for i in range(n_initial):
            if check_time() or n_calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = call_func(x)
            update(x, f)

        if best_x is None:
            x = np.zeros(dim)
            f = call_func(x)
            update(x, f)
            if best_x is None:
                return None

        # Phase 2: CMA-ES restarts
        # Use remaining budget for CMA-ES runs
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
            n_calls += used  # already tracked in call_func, but for safety
            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
