"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead simplex on the rotated Rastrigin f15."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        DEADLINE = start + 275.0  # safety margin well inside the 300 s limit

        # ---- global state ----
        best_x = np.full(dim, np.nan)
        best_f = np.inf

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def timed(func_call, x):
            f = func_call(np.asarray(x, dtype=float))
            if not np.isfinite(f):
                f = 1e12
            return float(f)

        def nm_simplex(func_call, x0, max_evals, alpha, gamma, rho):
            """Nelder-Mead; returns (best_x, best_f, evals_used)."""
            n = x0.shape[0]
            x0 = np.asarray(x0, dtype=float)
            x0 = np.clip(x0, -5.0, 5.0)
            simplex = np.empty((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                p = x0.copy()
                p[i] += alpha
                simplex[i + 1] = np.clip(p, -5.0, 5.0)

            vals = np.empty(n + 1)
            used = 0
            for i in range(n + 1):
                vals[i] = timed(func_call, simplex[i])
                used += 1

            # order vertices ascending
            order = np.argsort(vals)
            simplex = simplex[order]
            vals = vals[order]
            record(simplex[0], vals[0])

            while used < max_evals:
                centroid = simplex[1:].mean(axis=0)
                xr = centroid + gamma * (centroid - simplex[0])
                fr = timed(func_call, xr)
                used += 1

                if fr < vals[0]:
                    xe = centroid + gamma * (gamma * (centroid - simplex[0]))
                    fe = timed(func_call, xe)
                    used += 1
                    if fe < fr:
                        simplex[0] = np.clip(xe, -5.0, 5.0)
                        vals[0] = fe
                    else:
                        simplex[0] = np.clip(xr, -5.0, 5.0)
                        vals[0] = fr
                elif fr < vals[-1]:
                    simplex[0] = np.clip(xr, -5.0, 5.0)
                    vals[0] = fr
                else:
                    xc = centroid + rho * (simplex[0] - centroid)
                    fc = timed(func_call, xc)
                    used += 1
                    if fc < vals[0]:
                        simplex[0] = np.clip(xc, -5.0, 5.0)
                        vals[0] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            vals[i] = timed(func_call, simplex[i])
                            used += 1

                order = np.argsort(vals)
                simplex = simplex[order]
                vals = vals[order]
                record(simplex[0], vals[0])

                # convergence guard
                spread = np.max(np.abs(simplex - simplex[0]))
                if spread < 1e-6:
                    break

            return simplex[0].copy(), vals[0], used

        # ---- Phase 1: dense grid of 25 diverse starts, fixed small NM budget each ----
        starts = [np.zeros(dim)]
        for r in (3.0, 1.5):
            for sx in (-1.0, 0.0, 1.0):
                for sy in (-1.0, 0.0, 1.0):
                    for sz in (-1.0, 0.0, 1.0):
                        v = np.array([sx, sy, sz], dtype=float)
                        if np.dot(v, v) <= 3.0:
                            starts.append(r * v)
        starts = [np.clip(s, -5.0, 5.0) for s in starts]

        phase1 = 20
        for s in starts:
            if used_total := (budget - (len(starts) - starts.index(s)) * 0):
                pass
            if time.time() > DEADLINE:
                break
            if budget <= 0:
                break
            _x, _f, _u = nm_simplex(func, s, phase1, alpha=0.6, gamma=1.5, rho=0.5)
            budget -= _u

        # ---- Phase 2: refined NM from the best basin point, with the remaining budget ----
        if np.isfinite(best_f) and budget > 2 and time.time() < DEADLINE:
            _x, _f, _u = nm_simplex(func, best_x, budget, alpha=0.05, gamma=1.5, rho=0.5)
            budget -= _u

        if not np.isfinite(best_f):
            best_x = np.zeros(dim)
            best_f = timed(func, best_x)

        return best_x, float(best_f)
# EVOLVE-BLOCK-END
