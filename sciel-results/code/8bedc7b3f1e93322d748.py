import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Dense random initialization
        n_starts = 25
        starts = rng.uniform(-5, 5, (n_starts, dim))
        f_starts = np.empty(n_starts)
        for i in range(n_starts):
            if calls_used >= budget or time.time() > deadline:
                f_starts[i:] = np.inf
                break
            f_starts[i] = func(starts[i])
            calls_used += 1
            update_best(starts[i], f_starts[i])

        # Phase 2: Nelder-Mead refinement for top starts
        # Sort starts by fitness
        valid_idx = np.where(np.isfinite(f_starts))[0]
        if len(valid_idx) > 0:
            sorted_idx = valid_idx[np.argsort(f_starts[valid_idx])]
        else:
            sorted_idx = np.arange(n_starts)

        # Take top 5 starts for Nelder-Mead
        n_nm = min(5, len(sorted_idx))
        nm_starts = starts[sorted_idx[:n_nm]]
        nm_fits = f_starts[sorted_idx[:n_nm]]

        # Allocate budget for Nelder-Mead
        # Each NM run gets a fixed budget
        nm_budget_each = 30

        for i in range(n_nm):
            if calls_used >= budget or time.time() > deadline:
                break

            x0 = nm_starts[i].copy()
            f0 = nm_fits[i]

            # Run Nelder-Mead
            nm_best_x, nm_best_f, nm_calls = self._nelder_mead(
                func, x0, f0, nm_budget_each, dim, deadline
            )
            calls_used += nm_calls
            update_best(nm_best_x, nm_best_f)

        # Phase 3: CMA-ES from the best point found so far
        if calls_used < budget and best_x is not None:
            remaining = budget - calls_used
            if remaining > 20:
                cma_budget = min(remaining, 200)  # Cap CMA-ES budget
                cma_best_x, cma_best_f, cma_calls = self._cmaes(
                    func, best_x, 1.0, cma_budget, dim, deadline
                )
                calls_used += cma_calls
                update_best(cma_best_x, cma_best_f)

        # Phase 4: If budget remains, do a final local search with smaller steps
        if calls_used < budget and best_x is not None and time.time() < deadline:
            # Do a few more random perturbations around the best point
            n_extra = min(50, budget - calls_used)
            for _ in range(n_extra):
                if calls_used >= budget or time.time() > deadline:
                    break
                # Small perturbation
                step = 0.1 * (1.0 - (calls_used / budget))  # Decreasing step
                x_new = best_x + step * rng.standard_normal(dim)
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls_used += 1
                update_best(x_new, f_new)

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, max_calls, dim, deadline):
        """Nelder-Mead simplex method."""
        n = dim
        calls = 1  # f0 already evaluated
        best_x = x0.copy()
        best_f = f0

        # Create initial simplex
        simplex = np.empty((n + 1, n))
        simplex[0] = x0
        fvals = np.empty(n + 1)
        fvals[0] = f0

        # Create other points
        for i in range(1, n + 1):
            if calls >= max_calls or time.time() > deadline:
                break
            simplex[i] = x0.copy()
            simplex[i, i - 1] += 0.25
            simplex[i] = np.clip(simplex[i], -5, 5)
            fvals[i] = func(simplex[i])
            calls += 1
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = simplex[i].copy()

        if calls >= max_calls or time.time() > deadline:
            return best_x, best_f, calls

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        while calls < max_calls and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

            # Centroid of all but worst point
            centroid = np.mean(simplex[:n], axis=0)

            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[n])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls += 1
            if calls >= max_calls or time.time() > deadline:
                break
            if f_ref < best_f:
                best_f = f_ref
                best_x = x_ref.copy()

            if fvals[0] <= f_ref < fvals[n - 1]:
                # Accept reflection
                simplex[n] = x_ref
                fvals[n] = f_ref
            elif f_ref < fvals[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls += 1
                if calls >= max_calls or time.time() > deadline:
                    break
                if f_exp < best_f:
                    best_f = f_exp
                    best_x = x_exp.copy()
                if f_exp < f_ref:
                    simplex[n] = x_exp
                    fvals[n] = f_exp
                else:
                    simplex[n] = x_ref
                    fvals[n] = f_ref
            else:
                # Contraction
                if f_ref < fvals[n]:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (centroid - simplex[n])
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls += 1
                if calls >= max_calls or time.time() > deadline:
                    break
                if f_con < best_f:
                    best_f = f_con
                    best_x = x_con.copy()
                if f_con < min(f_ref, fvals[n]):
                    simplex[n] = x_con
                    fvals[n] = f_con
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls >= max_calls or time.time() > deadline:
                            break
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if fvals[i] < best_f:
                            best_f = fvals[i]
                            best_x = simplex[i].copy()
                    if calls >= max_calls or time.time() > deadline:
                        break

        return best_x, best_f, calls

    def _cmaes(self, func, x0, sigma0, max_calls, dim, deadline):
        """Standard CMA-ES implementation."""
        n = dim
        lam = max(4, int(4 + 2 * np.log(n)))
        mu = lam // 2

        # Weights
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_w = (w ** 2).sum()
        mu_eff = 1.0 / mu_w

        # Parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 1) / (mu_eff + 2) ** 2 - 1))
        chi_n = np.sqrt(n * (1 - 1 / (4 * n) - 1 / (21 * n ** 2)))

        # State
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        C_sqrt = np.eye(n)
        C_sqrt_inv = np.eye(n)

        calls = 0
        best_x = mean.copy()
        best_f = func(mean)
        calls = 1

        if calls >= max_calls:
            return best_x, best_f, calls

        while calls < max_calls:
            if time.time() > deadline:
                break

            # Sample
            Z = np.random.default_rng().standard_normal((lam, n))
            Y = Z @ C_sqrt
            X = mean[None, :] + sigma * Y
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.empty(lam)
            for i in range(lam):
                if calls >= max_calls:
                    break
                F[i] = func(X[i])
                calls += 1
            if calls < lam:
                F[calls:] = np.inf

            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()

            # Sort
            order = np.argsort(F)
            X_sorted = X[order]

            # Update mean
            mean_old = mean.copy()
            mean = mean + sigma * np.sum(w[:, None] * X_sorted[:mu], axis=0)

            # Update ps
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_w) / sigma * (C_sqrt_inv @ (mean - mean_old))

            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lam + 1)))) < (1 + 2 / (n + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * c_c * (mean - mean_old) / sigma

            # Update C
            delta_sig = hsig * (1 - c_c) + c_c
            coeff_c = 1 - c_mu + c_1 * (n + delta_sig - 1)
            C = coeff_c * C + c_mu * (np.outer(mean - mean_old, mean - mean_old) / sigma ** 2)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update C_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            B = eigvecs
            C_sqrt = (B * D) @ B.T
            C_sqrt_inv = (B / D) @ B.T

            # If sigma is too small, break
            if sigma < 1e-10:
                break

        return best_x, best_f, calls
