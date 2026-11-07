"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-restart and final local refinement for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, budget, max_time, rng, dim, sigma0=1.0):
        """
        CMA-ES implementation.
        Returns (best_x, best_f, calls_made).
        """
        calls = 0
        x = x0.copy()
        C = np.eye(dim)
        sigma = sigma0
        best_x = x.copy()
        best_f = float('inf')

        # Strategy parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        mu = lambda_ // 2  # number of parents
        # Weights
        weights = np.log(np.arange(mu, 0, -1) + 0.5) - np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        # Adaptation rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * ((mu_eff + 2) / (dim + 2) - 1) / ((dim + 1.3)**2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

        ps = np.zeros(dim)
        pc = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Initial evaluation
        if calls < budget and time.time() < max_time:
            f = func(x)
            calls += 1
            best_f = f
            best_x = x.copy()
        else:
            return best_x, best_f, calls

        # Main loop
        while calls < budget and time.time() < max_time:
            # Generate offspring
            offspring = []
            for i in range(lambda_):
                if calls >= budget or time.time() > max_time:
                    break
                z = rng.standard_normal(dim)
                # Use Cholesky of C for transformation
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                xi = x + sigma * (L @ z)
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                offspring.append((fi, xi))

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Select parents
            parents = [offspring[i][1] for i in range(mu)]
            parent_weights = weights

            # New mean
            x_old = x.copy()
            x = sum(w * xi for w, xi in zip(parent_weights, parents))

            # Update path accumulators
            z_bar = np.linalg.inv(L) @ (x - x_old) / sigma if sigma > 1e-15 else np.zeros(dim)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * z_bar
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chi_n < (1.4 + 2 / (dim + 1)) * 1.0)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x - x_old) / sigma

            # Update covariance
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            diff = np.array([xi - x_old for _, xi in parents])
            C = (1 + c1 * delta_hsig - c1 - cmu) * C \
                + c1 * (np.outer(pc, pc) + delta_hsig * C) \
                + cmu * sum(w * np.outer(d, d) for w, d in zip(parent_weights, diff))

            # Regularize C to ensure PSD
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Termination check: if sigma is very small or progress is negligible
            if sigma < 1e-10:
                break

        return best_x, best_f, calls

    def _nelder_mead(self, func, start_x, budget, max_time, rng, dim):
        """Nelder-Mead for final local refinement."""
        calls = 0
        simplex = [start_x.copy()]
        base_sigma = 0.1
        for i in range(dim):
            pt = start_x.copy()
            pt[i] += base_sigma
            pt = np.clip(pt, -5, 5)
            simplex.append(pt)

        simplex_f = []
        for pt in simplex:
            if calls >= budget or time.time() > max_time:
                break
            f = func(pt)
            calls += 1
            simplex_f.append(f)

        if len(simplex_f) < dim + 1:
            return start_x, min(simplex_f) if simplex_f else float('inf'), calls

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        shrink = 0.5

        best_f = min(simplex_f)
        best_x = simplex[np.argmin(simplex_f)].copy()

        max_iter = min(budget - calls, 80)
        for _ in range(max_iter):
            if calls >= budget or time.time() > max_time:
                break

            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            if simplex_f[-1] - simplex_f[0] < 1e-14:
                break

            centroid = np.mean(simplex[:-1], axis=0)
            worst_pt = simplex[-1]
            worst_f = simplex_f[-1]

            x_ref = centroid + alpha * (centroid - worst_pt)
            x_ref = np.clip(x_ref, -5, 5)
            if calls >= budget or time.time() > max_time:
                break
            f_ref = func(x_ref)
            calls += 1

            if f_ref < simplex_f[0]:
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                if calls >= budget or time.time() > max_time:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                    if f_ref < best_f:
                        best_f = f_ref
                        best_x = x_ref.copy()
                    break
                f_exp = func(x_exp)
                calls += 1
                if f_exp < f_ref:
                    x_new, f_new = x_exp, f_exp
                else:
                    x_new, f_new = x_ref, f_ref
                simplex[-1] = x_new
                simplex_f[-1] = f_new
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
            elif f_ref < simplex_f[-2]:
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()
            else:
                x_con = centroid + rho * (worst_pt - centroid)
                x_con = np.clip(x_con, -5, 5)
                if calls >= budget or time.time() > max_time:
                    break
                f_con = func(x_con)
                calls += 1
                if f_con < worst_f:
                    simplex[-1] = x_con
                    simplex_f[-1] = f_con
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()
                else:
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls >= budget or time.time() > max_time:
                            break
                        f = func(simplex[i])
                        calls += 1
                        simplex_f[i] = f
                        if f < best_f:
                            best_f = f
                            best_x = simplex[i].copy()

        return best_x, best_f, calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        max_time = start_time + 285.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        # Phase 1: Initial exploration with diverse points
        # Use stratified sampling to cover the space
        n_init = 20
        init_points = []
        for i in range(n_init):
            if calls >= budget or time.time() > max_time:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            calls += 1
            init_points.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Also evaluate center
        x0 = np.zeros(dim)
        fx0 = func(x0)
        calls += 1
        if fx0 < best_f:
            best_f = fx0
            best_x = x0.copy()

        # Sort initial points
        init_points.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES from top 2 starting points
        # Allocate budget: 70% for CMA-ES, 30% for local refinement
        cma_budget = int(budget * 0.70)
        nm_budget = budget - cma_budget

        # Select 2 diverse starting points
        starts = [best_x.copy()]
        # Find a diverse second start
        for fx, x in init_points:
            if np.linalg.norm(x - best_x) > 1.5:
                starts.append(x.copy())
                break
        if len(starts) < 2:
            starts.append(init_points[0][1].copy())

        cma_per_start = cma_budget // len(starts)

        for i, start_pt in enumerate(starts):
            if calls >= cma_budget or time.time() > max_time:
                break
            remaining = cma_budget - calls
            budget_this = min(cma_per_start, remaining)
            if budget_this < 20:
                break

            sigma0 = 1.0 if i == 0 else 0.5
            nx, nf, nc = self._cma_es(func, start_pt, budget_this, max_time, rng, dim, sigma0)
            calls += nc
            if nf < best_f:
                best_f = nf
                best_x = nx

        # Phase 3: Final local refinement with Nelder-Mead
        if calls < budget and time.time() < max_time:
            nm_remaining = budget - calls
            if nm_remaining >= 15:
                nx, nf, nc = self._nelder_mead(func, best_x, nm_remaining, max_time, rng, dim)
                calls += nc
                if nf < best_f:
                    best_f = nf
                    best_x = nx

        return best_x, best_f
# EVOLVE-BLOCK-END
