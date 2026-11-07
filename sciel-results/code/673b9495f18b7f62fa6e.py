"""Model-and-optimize approach for Rastrigin-rotated f15."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


class Optimizer:
    """
    Two-phase optimizer for Rastrigin-rotated f15:
    1. Sobol quasi-random sampling to map the landscape and identify basins
    2. CMA-ES refinement from the best identified basin
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
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            xmean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            local_best_f = np.inf
            local_best_x = x0.copy()

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

                pops = []
                fitvals = []
                for i in range(lam):
                    if not time_check():
                        break
                    z = rng.standard_normal(n)
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

                order = np.argsort(fitvals)
                pops = np.array([pops[i] for i in order])
                fitvals = np.array([fitvals[i] for i in order])

                xnew = np.sum(w[:, None] * pops[:mu], axis=0)

                try:
                    invsqrtC = np.linalg.inv(np.sqrtm(C))
                except Exception:
                    invsqrtC = np.eye(n)
                pc = (1 - cc) * pc + cc * invsqrtC @ (xnew - xmean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * invsqrtC @ (xnew - xmean) / sigma

                sigma *= np.exp(cs * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)

                C = (1 + c1 * (cc * (1 + cc) - 1) + cmu) * C
                C += c1 * np.outer(pc, pc)
                C += cmu * np.sum(w[:, None, None] * np.outer(pops[:mu] - xmean, pops[:mu] - xmean) / sigma ** 2, axis=0)

                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = (C + C.T) / 2

                xmean = xnew
                gen += 1

            return local_best_x, local_best_f

        # Phase 1: Sobol quasi-random sampling to map the landscape
        n_sobol = min(80, budget // 8)
        
        # Generate Sobol sequence points
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_sobol)
        # Scale to [-5, 5]
        sobol_points = sobol_points * 10.0 - 5.0
        
        # Evaluate all Sobol points
        for p in sobol_points:
            if not time_check():
                break
            eval_func(p)
        
        # Also evaluate origin
        if time_check():
            eval_func(np.zeros(dim))

        # Collect top candidates
        # We need to track which points were evaluated and their values
        # Since we don't store all points, let's just use the global best
        # and generate a few perturbations around it
        
        # Phase 2: CMA-ES refinement from best point
        if time_check() and best_f < np.inf:
            # Run CMA-ES from the best point found
            sigma0 = 1.5
            max_evals = min(200, budget - calls_made)
            bx, bf = cmaes(best_x.copy(), sigma0, max_evals)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
        
        # Phase 3: Fine-tuning with smaller sigma
        if time_check() and best_f < np.inf and budget - calls_made > 15:
            sigma0 = 0.2
            max_evals = min(50, budget - calls_made)
            bx, bf = cmaes(best_x.copy(), sigma0, max_evals)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
