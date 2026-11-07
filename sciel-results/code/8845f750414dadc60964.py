"""CMA-ES with restarts for Rastrigin-rotated f15."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES with multiple restarts for Rastrigin-rotated f15.
    Uses population-based evolution with covariance adaptation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        calls_made = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_check():
            return time.time() < deadline and calls_made < budget

        def cmaes(x0, sigma0, max_evals):
            """Run CMA-ES from x0 for at most max_evals function evaluations."""
            nonlocal calls_made, best_f, best_x
            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))  # population size
            mu = lam // 2
            # Logarithmic weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            # Learning rates
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            # Initial state
            xmean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Track best within this run
            local_best_f = np.inf
            local_best_x = x0.copy()

            # Evaluate initial point
            fx0 = eval_func(xmean)
            if fx0 < local_best_f:
                local_best_f = fx0
                local_best_x = xmean.copy()

            gen = 0
            while calls_made < budget and time.time() < deadline:
                if not time_check():
                    break
                if gen * lam > max_evals:
                    break

                # Generate population
                pops = []
                fitvals = []
                for i in range(lam):
                    if not time_check():
                        break
                    # Sample from N(xmean, sigma^2 * C)
                    z = rng.standard_normal(n)
                    # Transform using Cholesky of C
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(n)
                    y = xmean + sigma * L @ z
                    y = np.clip(y, -5.0, 5.0)
                    fy = eval_func(y)
                    pops.append(y)
                    fitvals.append(fy)
                    if fy < local_best_f:
                        local_best_f = fy
                        local_best_x = y.copy()

                if len(pops) < mu:
                    break

                # Sort by fitness
                order = np.argsort(fitvals)
                pops = np.array([pops[i] for i in order])
                fitvals = np.array([fitvals[i] for i in order])

                # Weighted recombination
                xnew = np.sum(w[:, None] * pops[:mu], axis=0)

                # Update path
                invsqrtC = np.linalg.inv(np.sqrtm(C)) if n <= 10 else np.eye(n)
                pc = (1 - cc) * pc + cc * invsqrtC @ (xnew - xmean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * invsqrtC @ (xnew - xmean) / sigma

                # Hessian-free rank-mu update
                v = invsqrtC @ (xnew - xmean) / sigma
                delta_sig = 1 + cs * (np.linalg.norm(ps) / chi_n - 1)
                sigma *= np.exp(cs * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)

                # Update C
                C = (1 + c1 * (cc * (1 + cc) - 1) + cmu) * C
                C += c1 * np.outer(pc, pc)
                C += cmu * np.sum(w[:, None, None] * np.outer(pops[:mu] - xmean, pops[:mu] - xmean) / sigma ** 2, axis=0)

                # Symmetrize and ensure positive definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = (C + C.T) / 2

                xmean = xnew

                gen += 1

            return local_best_x, local_best_f

        # Phase 1: Initial exploration with a few CMA-ES runs from different starts
        # Start from origin and a few random points
        start_points = [np.zeros(dim)]
        for _ in range(4):
            p = rng.uniform(-4, 4, dim)
            start_points.append(p)

        # Allocate budget across restarts
        n_restarts = 5
        evals_per_restart = max(50, budget // n_restarts)

        for i, sp in enumerate(start_points[:n_restarts]):
            if not time_check():
                break
            remaining = budget - calls_made
            if remaining < 20:
                break
            max_evals = min(evals_per_restart, remaining)
            sigma0 = 2.0 if i == 0 else 1.0
            bx, bf = cmaes(sp.copy(), sigma0, max_evals)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # Phase 2: Refinement around the best found point
        if time_check() and best_f < np.inf and budget - calls_made > 30:
            sigma0 = 0.5
            bx, bf = cmaes(best_x.copy(), sigma0, min(100, budget - calls_made))
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # Phase 3: Final fine-tuning with very small sigma
        if time_check() and best_f < np.inf and budget - calls_made > 10:
            sigma0 = 0.1
            bx, bf = cmaes(best_x.copy(), sigma0, min(50, budget - calls_made))
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
