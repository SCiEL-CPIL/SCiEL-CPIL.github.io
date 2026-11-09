"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive restarts for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        best_x = np.full(n, np.nan)
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            val = func(x)
            return val

        # CMA-ES implementation
        def cmaes(func_eval, x0, sigma0, max_iter, deadline_check):
            n = len(x0)
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            c_c = (1 + 3 / (n + 1)) / ((n + 1) / 2 + 3)
            c_sigma = (n + 1.7) / (n + 9 + 4 * n)
            c1 = 2 / ((n + 1.3) ** 2 + n)
            cmu = min(0.4 - c1, 2 / ((n + 2) ** 2 - n))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            mu = n // 2
            weights = np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            nu = 1 / (np.sum(weights) ** 2 - 1)

            fx = func_eval(x)
            best_local_x = x.copy()
            best_local_f = fx
            it = 0

            while it < max_iter:
                if deadline_check():
                    break

                # Generate lambda candidates
                lam = max(2 * n, 4)
                ys = np.random.default_rng(it).standard_normal((lam, n))
                # Use a fixed seed per iteration for reproducibility within a run
                # Actually, let's use a global rng for better mixing
                # We'll pass rng in or use a module-level one
                # For simplicity, let's use np.random with a fixed seed strategy
                # Better: use a persistent rng
                pass  # placeholder, we'll restructure

            return best_local_x, best_local_f

        # Let's implement a cleaner CMA-ES
        rng = np.random.default_rng(42)

        def cmaes_run(x0, sigma0, max_iter, deadline_check):
            nonlocal call_count, best_x, best_f
            n = len(x0)
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            c_c = (1 + 3 / (n + 1)) / ((n + 1) / 2 + 3)
            c_sigma = (n + 1.7) / (n + 9 + 4 * n)
            c1 = 2 / ((n + 1.3) ** 2 + n)
            cmu = min(0.4 - c1, 2 / ((n + 2) ** 2 - n))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            mu = n // 2
            weights = np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            nu = 1 / (np.sum(weights) ** 2 - 1)
            lam = max(2 * n, 4)

            fx = safe_func(x)
            if not np.isinf(fx):
                best_local_x = x.copy()
                best_local_f = fx
            else:
                best_local_x = x.copy()
                best_local_f = fx

            it = 0
            while it < max_iter and call_count < budget:
                if deadline_check():
                    break

                # Generate candidates
                ys = rng.standard_normal((lam, n))
                xs = np.array([x + sigma * y for y in ys])
                xs = np.clip(xs, -5, 5)

                # Evaluate
                fs = np.array([safe_func(xi) for xi in xs])
                if call_count >= budget:
                    break

                # Update best
                idx = np.argmin(fs)
                if fs[idx] < best_local_f:
                    best_local_x = xs[idx].copy()
                    best_local_f = fs[idx]
                    if best_local_f < best_f:
                        best_f = best_local_f
                        best_x = best_local_x.copy()

                # Sort
                order = np.argsort(fs)
                ys_sorted = ys[order]
                xs_sorted = xs[order]
                fs_sorted = fs[order]

                # Check for stagnation (no improvement in last 100 iterations)
                # Simple check: if best hasn't improved in 50 iterations, break
                # For now, just run until max_iter or budget

                # Update mean
                x_new = np.sum(weights[:, None] * ys_sorted[:mu], axis=0) * sigma + x
                x = np.clip(x_new, -5, 5)

                # Update covariance
                yw = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)
                yw = yw / sigma

                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * yw
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * yw

                if hsig:
                    C = (1 + c1 * cmu - c1) * C + (c1 * cmu) * np.outer(yw, yw) + (c1 * cmu) * (np.sum(weights) * nu * np.outer(pc, pc) - c1 * C)
                else:
                    C = (1 + c1 * cmu - c1) * C + (c1 * cmu) * np.outer(yw, yw)

                # Re-orthogonalize periodically
                if it % 10 == 0:
                    D2, B2 = np.linalg.eigh(C)
                    D2 = np.sqrt(np.clip(D2, 1e-12, None))
                    C = np.outer(B2, B2.T) * (D2[:, None] * D2[None, :])
                    B = B2
                    D = D2

                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))

                it += 1

            return best_local_x, best_local_f

        def deadline_check():
            return time.time() > deadline

        # Multi-start strategy
        # Divide budget among several starts
        num_starts = 3
        budget_per_start = budget // num_starts
        # Give a bit more to later starts if earlier ones fail
        # Actually, let's do: first start gets 40%, second 30%, third 30%
        budgets = [int(budget * 0.4), int(budget * 0.3), int(budget * 0.3)]
        budgets[-1] = budget - budgets[0] - budgets[1]  # adjust last to use remaining

        for i in range(num_starts):
            if call_count >= budget or deadline_check():
                break

            # Choose starting point
            if i == 0:
                x0 = rng.uniform(-4, 4, n)
                sigma0 = 1.5
            elif i == 1:
                # Start near current best with perturbation
                if not np.isnan(best_x[0]):
                    x0 = np.clip(best_x + rng.standard_normal(n) * 0.5, -5, 5)
                else:
                    x0 = rng.uniform(-4, 4, n)
                sigma0 = 0.5
            else:
                # Another random start
                x0 = rng.uniform(-4, 4, n)
                sigma0 = 1.0

            max_iter = budgets[i]
            cmaes_run(x0, sigma0, max_iter, deadline_check)

        # Final refinement with Nelder-Mead if time allows
        if not np.isnan(best_x[0]) and call_count < budget and not deadline_check():
            try:
                from scipy.optimize import minimize
                # Use remaining budget for NM
                remaining = budget - call_count
                if remaining > 10:
                    res = minimize(safe_func, best_x, method='Nelder-Mead',
                                   options={'maxiter': remaining, 'xatol': 1e-8, 'fatol': 1e-8})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
            except:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
