"""Rewritten optimizer: full-covariance CMA-ES with PCA-seeded covariance."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Full-covariance CMA-ES with PCA-seeded covariance for BBOB f18
    (Schaffers F7 ill-conditioned).

    Strategy:
      * Adapt the search distribution to local curvature/scaling (full C).
      * Seed C from the PCA of the best early samples so the anisotropic
        search aligns with the narrow funnel immediately (early progress).
      * Corrected covariance update (proper path-history hsig factor).
      * Stagnation-triggered restart that re-seeds PCA from current bests.
      * A final refinement tail that shrinks sigma to squeeze out the last
        bits of accuracy.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0  # safety margin before the 330s kill

        dim = self.dim
        total_budget = self.budget

        # ---- CMA-ES parameters (standard) ----
        lam = max(4, int(4 + 3 * math.log(dim)))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 2.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        ds = 1.0 + 2.0 * max(0.0, math.sqrt((mu - 1.0) / (dim + 1.0)) - 1.0) + cs

        best_x = np.zeros(dim)
        best_f = float("inf")
        calls = 0

        def time_ok():
            return (time.time() - start_time) < (deadline - start_time) - 3.0

        def evaluate(x):
            nonlocal best_x, best_f, calls
            try:
                f = float(func(x))
            except Exception:
                f = float("inf")
            calls += 1
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float, copy=True)
            return f

        # ---- helper: PCA seed from a set of (x, f) points ----
        def pca_seed(points, x, f):
            """Return (mean, C, sigma) seeded from PCA of the best points."""
            n = len(points)
            if n >= 2:
                pts = np.array([p[0] for p in points])
                m = pts.mean(axis=0)
                centered = pts - m
                # covariance of the best points
                C = (centered.T @ centered) / (n - 1)
                # regularize for PD
                C = C + 1e-6 * np.eye(dim)
                try:
                    e_vals, e_vecs = np.linalg.eigh(C)
                    e_vals = np.clip(e_vals, 1e-12, None)
                    # scale so total spread is comparable to sigma0
                    scale = (2.0 ** 2) / max(np.trace(C), 1e-12)
                    C = (e_vecs @ (e_vals * scale * e_vals) @ e_vecs.T) / e_vals.sum()
                    C = e_vecs @ np.diag(e_vals * scale) @ e_vecs.T
                    C = C + 1e-9 * np.eye(dim)
                    sigma = 1.5
                except Exception:
                    m = x.copy()
                    C = np.eye(dim)
                    sigma = 2.0
            else:
                m = x.copy()
                C = np.eye(dim)
                sigma = 2.0
            return m, C, sigma

        # Keep a record of (x, f) for PCA seeding across the whole run.
        all_points = []

        # ---- main search loop with restarts ----
        run = 0
        while calls < total_budget and time_ok():
            run += 1
            run_calls = 0
            run_no_imp = 0
            run_best_f = float("inf")

            # Initialize state
            if run == 1:
                mean = np.zeros(dim)
                C = np.eye(dim)
                sigma = 2.0
            else:
                # re-seed from PCA of best points found so far
                mean, C, sigma = pca_seed(all_points, best_x, best_f)

            ps = np.zeros(dim)
            pc = np.zeros(dim)
            try:
                chol = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except Exception:
                C = np.eye(dim)
                chol = np.eye(dim)

            # ---- inner CMA-ES loop ----
            while calls < total_budget and run_calls < 250 and time_ok():
                # sample population
                z = np.random.randn(lam, dim)
                pop = mean[None, :] + sigma * (z @ chol)
                pop = np.clip(pop, -5.0, 5.0)

                fvals = np.empty(lam)
                for i in range(lam):
                    if calls >= total_budget or not time_ok():
                        fvals[i] = float("inf")
                        continue
                    fvals[i] = evaluate(pop[i])
                    run_calls += 1
                    if fvals[i] < run_best_f:
                        run_best_f = fvals[i]
                        run_no_imp = 0
                    else:
                        run_no_imp += 1
                    if np.isfinite(fvals[i]):
                        all_points.append((pop[i].copy(), fvals[i]))

                order = np.argsort(fvals)
                spop = pop[order]
                sfvals = fvals[order]

                # weighted mean
                new_mean = np.sum(w[:, None] * spop[:mu], axis=0)

                # evolution paths
                norm_chol = np.linalg.norm(chol, "fro")
                ps = (1.0 - cs) * ps + math.sqrt(cs * (2.0 - cs) * mueff) / max(norm_chol, 1e-15) * (
                    chol.T @ (new_mean - mean) / sigma
                )
                ccs = cs * (dim + 1)
                hsig = 1.0 if (
                    np.linalg.norm(ps) / math.sqrt(1.0 - (1.0 - cs) ** ((calls + 1) / ccs))
                    < (1.4 + 2.0 / (dim + 1)) * math.sqrt((dim + 1) / (dim - 1))
                ) else 0.0
                pc = (1.0 - cc) * pc + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * (new_mean - mean) / sigma

                # covariance update (correct hsig factor)
                C = (1.0 + c1 * hsig * (cc - cc + 1.0) + cmu - c1) * C \
                    + c1 * (np.outer(pc, pc) - C) \
                    + cmu / (sigma ** 2) * np.sum(
                        w[:, None, None] * np.outer(spop[:mu] - mean, spop[:mu] - mean), axis=0
                    )
                C = 0.5 * (C + C.T)
                try:
                    chol = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                except Exception:
                    C = np.eye(dim)
                    chol = np.eye(dim)
                    sigma = 1.0

                # sigma update
                sigma = sigma * math.exp(cs / ds * (np.linalg.norm(ps) / math.sqrt(mueff) - 1.0))
                sigma = min(sigma, 50.0)
                mean = new_mean
                mean = np.clip(mean, -5.0, 5.0)

                # stagnation restart
                if run_no_imp > 90 and run_calls > 40:
                    break

        # ---- refinement tail: shrink sigma and polish the best point ----
        if calls < total_budget and time_ok() and np.isfinite(best_f):
            tail_budget = total_budget - calls
            # local refinement via small random perturbations (1+1 style)
            cur_x = best_x.copy()
            cur_f = best_f
            step = 0.5
            while calls < total_budget and time_ok() and step > 1e-4:
                cand = np.clip(cur_x + step * np.random.randn(dim), -5.0, 5.0)
                f = evaluate(cand)
                if f < cur_f:
                    cur_x = cand
                    cur_f = f
                    step = min(step * 1.1, 2.0)
                else:
                    step *= 0.7

        return best_x, best_f
# EVOLVE-BLOCK-END
