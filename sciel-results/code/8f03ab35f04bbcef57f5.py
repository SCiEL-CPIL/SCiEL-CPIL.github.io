"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5: lattice global search + adaptive-restart CMA-ES + NM polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Rastrigin-aware optimizer: lattice sampling + adaptive-restart CMA-ES + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    # ------------------------------------------------------------------ #
    #  CMA-ES (self-contained, no external dependency)
    # ------------------------------------------------------------------ #
    def _cma(self, func, x0, sigma0, budget, maxfev, deadline, dim):
        x0 = np.asarray(x0, dtype=float).copy()
        n = dim
        lam = 4 + int(2 * np.sqrt(n))
        mu = lam // 2
        # recombination weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / w.sum()
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        # eigen decomposition cache
        C_eigvals = np.ones(n)
        C_eigvecs = np.eye(n)
        damps = 1.0
        # precompute sqrt of C via eig
        C_eigvals = np.ones(n)
        C_eigvecs = np.eye(n)

        B = C_eigvecs
        D = np.sqrt(np.clip(C_eigvals, 1e-12, None))
        minfev = 1.0 / (1e-12 + 1e-8 / sigma0)
        minfev = max(minfev, 0.5 * (n + 3) ** 2 / (n + 1))
        minfev = max(minfev, 2.0 * n)

        sigma = sigma0
        x = x0.copy()
        count_eval = 0
        best_f = np.inf
        best_x = x0.copy()

        while count_eval < budget and count_eval < maxfev and time.time() < deadline:
            # sample
            z = np.random.standard_normal((lam, n))
            # apply C via eig: B @ diag(D) @ z
            y = (z * D) @ B.T
            pop = np.array([x + sigma * yi for yi in y])
            # evaluate
            fvals = np.empty(lam)
            for k in range(lam):
                if count_eval >= budget or time.time() >= deadline:
                    fvals[k] = np.inf
                    continue
                fx = float(func(np.clip(pop[k], -5.0, 5.0)))
                count_eval += 1
                fvals[k] = fx
                if fx < best_f:
                    best_f = fx
                    best_x = pop[k].copy()
            order = np.argsort(fvals)
            # recombination
            y_w = np.zeros(n)
            for k in range(mu):
                y_w += w[k] * (pop[order[k]] - x) / sigma
            x_new = x + sigma * y_w
            # sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (
                C_eigvecs.T @ (D * (C_eigvecs @ y_w))
            )
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (count_eval / (c_sigma * (n + 1)))) < (1.4 + 1 / c_sigma) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            # step-size control
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x) / sigma
            # update C
            v = C_eigvecs @ (D * (C_eigvecs @ p_c))
            C = ((1 + c_1 * (2 - c_1 - c_mu) - c_mu) * C
                 + c_1 * np.outer(v, v)
                 + c_mu * sum(w[k] * np.outer(C_eigvecs @ (D * (C_eigvecs @ (pop[order[k]] - x) / sigma)),
                                             C_eigvecs @ (D * (C_eigvecs @ (pop[order[k]] - x) / sigma)))
                              for k in range(mu)))
            # recompute eig (every step is fine for n=5)
            C_eigvals, C_eigvecs = np.linalg.eigh(C)
            C_eigvals = np.clip(C_eigvals, 1e-12, None)
            D = np.sqrt(C_eigvals)
            # stagnation / termination
            x = x_new
            if sigma < 1e-12 or count_eval > 20 * minfev * (n + 3) ** 2:
                break
            # convergence
            if (np.linalg.norm(x_new - x0) / max(sigma, 1e-12)) ** 2 < 2 * n * (1e-8) ** 2:
                break
        return best_x, best_f, count_eval

    # ------------------------------------------------------------------ #
    #  Nelder-Mead polish
    # ------------------------------------------------------------------ #
    def _nelder_mead(self, func, x0, budget, maxfev, deadline, dim):
        n = dim
        x0 = np.asarray(x0, dtype=float).copy()
        # build simplex
        simplex = [x0.copy()]
        for i in range(n):
            p = x0.copy()
            step = 0.05 if abs(x0[i]) <= 1e-8 else 0.05 * abs(x0[i])
            p[i] += step
            simplex.append(p)
        fvals = np.empty(n + 1)
        count = 0

        def eval_pt(p):
            nonlocal count
            if count >= budget or time.time() >= deadline:
                return np.inf
            v = float(func(np.clip(p, -5.0, 5.0)))
            count += 1
            return v

        for k in range(n + 1):
            fvals[k] = eval_pt(simplex[k])
            if count >= budget or time.time() >= deadline:
                break

        best_f = np.inf
        best_x = x0.copy()
        for k in range(n + 1):
            if fvals[k] < best_f:
                best_f = fvals[k]
                best_x = simplex[k].copy()

        max_iter = max(1, (budget - count) // (n + 1))
        for _ in range(max_iter):
            if count >= budget or time.time() >= deadline:
                break
            order = np.argsort(fvals)
            simplex = [simplex[order[k]] for k in range(n + 1)]
            fvals = fvals[order]
            # centroid excluding worst
            centroid = np.mean(simplex[:n], axis=0)
            # reflect
            alpha = 1.0
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = eval_pt(xr)
            if count >= budget or time.time() >= deadline:
                break
            if fvals[0] <= fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            elif fr < fvals[0]:
                # expand
                gamma = 2.0
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = eval_pt(xe)
                if count >= budget or time.time() >= deadline:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            else:
                # contract
                rho = 0.5
                if fr < fvals[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                else:
                    xc = centroid - rho * (simplex[-1] - centroid)
                fc = eval_pt(xc)
                if count >= budget or time.time() >= deadline:
                    break
                if fc < max(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # shrink
                    for k in range(1, n + 1):
                        simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                        fvals[k] = eval_pt(simplex[k])
                        if count >= budget or time.time() >= deadline:
                            break
            # track best
            for k in range(n + 1):
                if fvals[k] < best_f:
                    best_f = fvals[k]
                    best_x = simplex[k].copy()
        return best_x, best_f, count

    # ------------------------------------------------------------------ #
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)
        call_count = [0]
        best_x = np.zeros(dim)
        best_f = np.inf

        def do_call(x):
            nonlocal best_f, best_x
            if call_count[0] >= budget:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = float(func(x_clipped))
            call_count[0] += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def time_ok():
            return time.time() < deadline

        # ---------- Phase 1: Lattice-aware global search ----------
        structured_points = [np.zeros(dim)]
        for i in range(dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                p = np.zeros(dim)
                p[i] = val
                structured_points.append(p)
        for i in range(dim):
            for j in range(i + 1, dim):
                for vi in [-1.0, 1.0]:
                    for vj in [-1.0, 1.0]:
                        p = np.zeros(dim)
                        p[i] = vi
                        p[j] = vj
                        structured_points.append(p)

        global_budget = min(120, budget // 4)
        num_random = max(0, global_budget - len(structured_points))
        all_points = structured_points[:]
        if num_random > 0:
            random_points = rng.uniform(-5, 5, (num_random, dim))
            all_points.extend(random_points)

        for p in all_points:
            if not time_ok() or call_count[0] >= global_budget:
                break
            do_call(p)

        # ---------- Phase 2: Adaptive-restart CMA-ES ----------
        # Track best and restart with increasing jump size from the current best
        # when a run stagnates (no improvement for `stagnation_limit` evaluations).
        remaining = budget - call_count[0]
        if remaining > 20 and time_ok():
            # We'll run CMA-ES in a loop. Each run gets a budget; on stagnation we
            # restart from the global best with a larger jump.
            cma_total_budget = remaining
            max_restarts = 4
            jump_base = 1.5
            # We implement a simple outer loop: run CMA-ES with a sub-budget,
            # check if it improved the global best; if not, restart with a jump.
            sub_budgets = [int(cma_total_budget * 0.5), int(cma_total_budget * 0.25),
                           int(cma_total_budget * 0.15), int(cma_total_budget * 0.10)]
            # ensure sum <= cma_total_budget
            while sum(sub_budgets) > cma_total_budget:
                sub_budgets[-1] = max(10, sub_budgets[-1] - 10)
            while sum(sub_budgets) < cma_total_budget and sub_budgets[-1] < cma_total_budget:
                sub_budgets[-1] += 10

            for r in range(max_restarts):
                if not time_ok() or call_count[0] >= budget:
                    break
                alloc = min(sub_budgets[r], budget - call_count[0])
                if alloc < 15:
                    break
                # start point: global best, but with a jump on restart
                if r == 0:
                    x0 = best_x.copy()
                    sigma0 = 1.0
                else:
                    # increasing jump from current best
                    jump = jump_base * (r + 1)
                    x0 = best_x + rng.normal(0, jump, dim)
                    x0 = np.clip(x0, -5.0, 5.0)
                    sigma0 = max(0.5, jump * 0.5)
                # run CMA-ES
                xb, fb, used = self._cma(do_call, x0, sigma0, alloc, alloc, deadline, dim)
                if fb < best_f:
                    best_f = fb
                    best_x = xb.copy()

        # ---------- Phase 3: Nelder-Mead polish ----------
        remaining = budget - call_count[0]
        if remaining > 10 and time_ok():
            xb, fb, used = self._nelder_mead(do_call, best_x, remaining, remaining, deadline, dim)
            if fb < best_f:
                best_f = fb
                best_x = xb.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
