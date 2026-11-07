"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid CMA-ES with multi-start and local coordinate-descent polish.
    Uses a single global RNG and strict budget/time guards.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin: evaluator kills at 330s. We aim to finish well before.
        time_limit = 310.0

        # CMA-ES parameters for dim=5
        lam = 4 + int(3 * np.log(dim))  # lam = 8 for dim=5
        mu = lam // 2  # mu = 4

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)

        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = (2 * (mueff - 2 + 2 / mueff)) / ((dim + 2)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))

        # Global state
        rng = np.random.default_rng(42)
        total_evals = 0
        best_x_global = None
        best_f_global = np.inf

        def record(x, f):
            nonlocal best_x_global, best_f_global
            if f < best_f_global:
                best_f_global = f
                best_x_global = x.copy()

        def check_time():
            return time.time() - start_time < time_limit

        def check_budget():
            return total_evals < budget

        def cmaes_local(x0, sigma0, max_evals):
            """Run CMA-ES from x0 with sigma0, using at most max_evals evaluations."""
            nonlocal total_evals
            m = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)

            # Initial evaluation
            fx0 = func(m)
            total_evals += 1
            record(m, fx0)

            if max_evals <= 1 or not check_time() or not check_budget():
                return

            while total_evals + lam <= budget and check_time():
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
                total_evals += lam

                idx_best = np.argmin(fs)
                record(xs[idx_best], fs[idx_best])

                if not check_time() or total_evals >= budget:
                    break

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

                # Convergence check
                if sigma < 1e-10:
                    break

        def local_polish(x0, max_evals):
            """Simple coordinate descent polish."""
            nonlocal total_evals
            best_x = x0.copy()
            best_f = func(best_x)
            total_evals += 1
            record(best_x, best_f)

            if max_evals <= 1 or not check_time() or not check_budget():
                return

            step = 0.1
            for _ in range(max_evals - 1):
                if not check_time() or total_evals >= budget:
                    break

                improved = False
                for d in range(dim):
                    for sign in [1, -1]:
                        if not check_time() or total_evals >= budget:
                            break
                        x_try = best_x.copy()
                        x_try[d] += sign * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        total_evals += 1
                        if f_try < best_f:
                            best_x = x_try
                            best_f = f_try
                            improved = True
                            record(best_x, best_f)

                if not improved:
                    step *= 0.5
                    if step < 1e-8:
                        break
                else:
                    step = max(step * 1.2, 0.01)

        # Multi-start strategy
        # Start 1: Origin (common shift for BBOB)
        cmaes_local(np.zeros(dim), 2.0, 200)

        # Start 2-6: Random points
        for i in range(5):
            if not check_time() or total_evals >= budget - 10:
                break
            x0 = rng.uniform(-4, 4, size=dim)
            cmaes_local(x0, 1.5, 60)

        # Final polish with remaining budget
        if best_x_global is not None:
            remaining = budget - total_evals
            if remaining > 10 and check_time():
                local_polish(best_x_global, remaining)

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
