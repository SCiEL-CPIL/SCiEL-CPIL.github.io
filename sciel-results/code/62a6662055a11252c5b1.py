"""Rewritten optimizer for BBOB f19 (Griewank-Rosenbrock) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start CMA-ES with adaptive restarts and a final Nelder-Mead polish
    for BBOB f19 (Griewank-Rosenbrock) on [-5,5]^5.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, max_calls, rng, deadline, global_best_f):
        """
        Run CMA-ES from x0. Stops when max_calls reached, deadline hit, or
        stagnation detected. Returns (best_x, best_f, calls_used).
        """
        dim = self.dim
        # Population size and weights
        lam = 4 + int(3 * np.log(dim))  # 8 for dim=5
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # CMA-ES parameters
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x = np.array(x0, dtype=float)
        sigma = float(sigma0)
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        calls_used = 0
        best_x = x.copy()
        best_f = func(x)
        calls_used += 1

        if best_f < global_best_f:
            global_best_f = best_f

        no_improve = 0
        stagnation_limit = 15  # generations without improvement before restart

        for gen in range(10000):
            if calls_used >= max_calls or time.time() >= deadline:
                break

            # Check stagnation
            if no_improve >= stagnation_limit:
                break

            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            # Sample population
            Z = rng.standard_normal((lam, dim))
            pop = x + sigma * (Z @ L.T)
            pop = np.clip(pop, -5.0, 5.0)

            # Evaluate
            fits = np.empty(lam)
            for i in range(lam):
                if calls_used >= max_calls or time.time() >= deadline:
                    break
                fits[i] = func(pop[i])
                calls_used += 1
            else:
                pass  # all evaluated
            if calls_used >= max_calls or time.time() >= deadline:
                # Only use the fits we actually computed
                lam_actual = calls_used - (calls_used - lam)  # this is tricky, let's just break
                # Actually, let's restructure: if we can't evaluate all, break
                # For simplicity, if we ran out of budget mid-population, break
                break

            # Update best
            idx_min = np.argmin(fits)
            if fits[idx_min] < best_f:
                best_f = fits[idx_min]
                best_x = pop[idx_min].copy()
                no_improve = 0
                if best_f < global_best_f:
                    global_best_f = best_f
            else:
                no_improve += 1

            # Sort
            sorted_idx = np.argsort(fits)
            y_sorted = pop[sorted_idx]

            # Weighted mean
            m_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

            # Update paths
            yw = (m_new - x) / sigma
            pc = (1 - cs) * pc + cs * yw
            ps = (1 - cs) * ps + cs * np.sign(yw)

            # Sigma update
            sigma *= np.exp(cs / chi * (np.linalg.norm(pc) / chi - 1))
            sigma = np.clip(sigma, 1e-12, 100.0)

            # Covariance update
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi
            delta_hsig = (1.0 if hsig else 0.0) * (1 - cs) * cs

            C = (1 + c1 + cmu - c1 - cmu) * C
            C += c1 * (np.outer(yw, yw) + (1.0 if hsig else 0.0) * cs * (1 - cs) - C)

            if cmu > 0:
                diff = y_sorted[:mu] - x
                C += cmu * np.sum(weights[:, None] * np.outer(diff, diff) / (sigma ** 2), axis=0)

            C = (C + C.T) / 2

            # Ensure PD
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += (1e-12 - np.min(eigvals) + 1e-12) * np.eye(dim)

            x = m_new

        return best_x, best_f, calls_used

    def _nelder_mead(self, func, x0, max_calls, rng, deadline):
        """
        Simple Nelder-Mead implementation with call counting.
        """
        dim = self.dim
        if max_calls < dim + 2:
            return x0, func(x0), 1

        # Initial simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(dim):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += 0.05

        fvals = np.array([func(simplex[i]) for i in range(dim + 1)])
        calls_used = dim + 1

        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        for _ in range(max_calls - calls_used):
            if time.time() >= deadline:
                break

            # Sort
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            best_f = fvals[0]
            worst_idx = dim
            centroid = np.mean(simplex[:dim], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[worst_idx])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = func(x_r)
            calls_used += 1

            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = func(x_e)
                calls_used += 1
                if f_e < f_r:
                    simplex[worst_idx] = x_e
                    fvals[worst_idx] = f_e
                else:
                    simplex[worst_idx] = x_r
                    fvals[worst_idx] = f_r
            elif f_r < fvals[dim - 1]:
                simplex[worst_idx] = x_r
                fvals[worst_idx] = f_r
            else:
                # Contraction
                if f_r < fvals[worst_idx]:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (simplex[worst_idx] - centroid)
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = func(x_c)
                calls_used += 1
                if f_c < f_r:
                    simplex[worst_idx] = x_c
                    fvals[worst_idx] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls_used += 1
                    if time.time() >= deadline:
                        break

            if calls_used >= max_calls:
                break

            # Check convergence
            if np.max(fvals) - np.min(fvals) < 1e-12:
                break

        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx], calls_used

    def __call__(self, func):
        deadline = time.time() + 320.0  # Safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        global_best_x = np.zeros(dim)
        global_best_f = np.inf
        calls_remaining = budget

        # Multi-start strategy with adaptive budget allocation
        # Start 1: Origin with large sigma
        # Start 2: Random point with medium sigma
        # Start 3: Random point with small sigma
        # Final: Nelder-Mead polish

        starts = [
            (np.zeros(dim), 2.0),
            (rng.uniform(-3, 3, dim), 1.5),
            (rng.uniform(-4, 4, dim), 1.0),
            (rng.uniform(-4, 4, dim), 0.5),
        ]

        # Allocate budget: 40%, 25%, 20%, 10%, 5% for polish
        allocs = [0.40, 0.25, 0.20, 0.10, 0.05]

        for i, (x0, sigma0) in enumerate(starts):
            if calls_remaining <= 0 or time.time() >= deadline:
                break

            alloc = int(calls_remaining * allocs[i])
            if alloc < 10:
                alloc = calls_remaining

            bx, bf, used = self._cmaes(func, x0, sigma0, alloc, rng, deadline, global_best_f)
            calls_remaining -= used

            if bf < global_best_f:
                global_best_f = bf
                global_best_x = bx.copy()

            # Early stop if very good
            if global_best_f < 1e-4:
                break

        # Final polish with Nelder-Mead
        if calls_remaining > 5 and time.time() < deadline:
            px, pf, pused = self._nelder_mead(func, global_best_x, calls_remaining, rng, deadline)
            if pf < global_best_f:
                global_best_f = pf
                global_best_x = px.copy()

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
