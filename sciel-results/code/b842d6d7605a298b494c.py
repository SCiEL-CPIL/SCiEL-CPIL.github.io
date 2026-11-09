"""Portfolio optimizer for BBOB f1 (Sphere) on [-5,5]^dim.

Strategy:
  1. A small deterministic global net (LHS + random + origin) to seed local solvers.
  2. CMA-ES from the best global point with a generous budget (adapts to
     rotation/shift via covariance adaptation).
  3. L-BFGS-B polish from several diverse starts.
  4. A final CMA-ES squeeze with the remaining budget.

The best point found is always returned.  A hard wall-clock deadline keeps
the process well inside the 330 s kill limit.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start (LHS + random) with CMA-ES / L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        # Safety margin so the process is never killed at 300/330s.
        deadline = t0 + 280.0

        def time_ok():
            return time.time() < deadline

        def wrap(z):
            x = np.clip(np.asarray(z, dtype=float).reshape(-1), -5.0, 5.0)
            try:
                return float(func(x))
            except Exception:
                return 1e30

        rng = np.random.default_rng(0)

        # Reduced initial exploration for faster early progress (AOCC rewards early improvement)
        # Use a smaller, more strategic set of initial points
        n_lhs = 8
        lhs = np.empty((n_lhs, dim))
        for j in range(dim):
            order = rng.permutation(n_lhs)
            lhs[:, j] = -5.0 + 10.0 * (order + 0.5) / n_lhs

        n_rand = 6
        rand = rng.uniform(-5.0, 5.0, size=(n_rand, dim))

        origin = np.zeros(dim)

        candidates = np.vstack([lhs, rand, origin[None, :]])

        # Order candidates so "central / easy" points are tried first
        scores = np.linalg.norm(candidates, axis=1)
        order = np.argsort(scores)
        candidates = candidates[order]

        # Evaluate all initial candidates to find the best starting point
        best_x = None
        best_f = float('inf')
        for cand in candidates:
            f_val = wrap(cand)
            if f_val < best_f:
                best_f = f_val
                best_x = cand
        budget_left = budget - len(candidates)

        # ------------------------------------------------------------------
        # Phase A: CMA-ES from the best of the global net (global basin finder).
        # Give CMA-ES a larger share of the budget since it's the most effective method
        # ------------------------------------------------------------------
        try:
            from cma import CMA
            if budget_left > 50:
                # Allocate up to 60% of remaining budget to first CMA-ES run
                cma_budget = int(min(350, max(100, budget_left * 0.60)))
                res = CMA(evolve=wrap, x0=best_x.tolist(),
                          sigma0=1.5,
                          bounds=[(-5.0,) * dim, (5.0,) * dim],
                          maxfevals=cma_budget,
                          seed=1,
                          verbose=-9, quiet=True,
                          pickle=False,
                          toff=0, cmaes_tolerances={
                              'tolask'  : 0.0,
                              'tolfun'  : 1e-14,
                              'tolx'    : 0.0,
                              'tolopt'  : 0.0,
                          })
                budget_left -= int(res.evaluations)
                if res.fbest <= best_f:
                    best_f = float(res.fbest)
                    best_x = np.array(res.xbest, dtype=float).reshape(dim)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Phase B: fast L-BFGS-B polish from diverse starts.
        # ------------------------------------------------------------------
        try:
            from scipy.optimize import minimize
            polish_starts = [best_x]
            # Add diverse global candidates from initial exploration
            for idx in [0, len(candidates) // 3, 2 * len(candidates) // 3, len(candidates) - 1]:
                if idx < len(candidates):
                    polish_starts.append(candidates[idx])
            # De-duplicate close starts.
            kept = []
            for s in polish_starts:
                if all(np.linalg.norm(s - k) > 1e-6 for k in kept):
                    kept.append(s)
            polish_starts = kept

            for s in polish_starts:
                if budget_left <= 5 or not time_ok():
                    break
                try:
                    r = minimize(wrap, s, method='L-BFGS-B',
                                 bounds=[(-5.0, 5.0)] * dim,
                                 options={'maxiter': 300,
                                          'maxfun': max(5, min(150, budget_left))})
                    budget_left -= int(r.nfev)
                    if r.fun <= best_f:
                        best_f = float(r.fun)
                        best_x = np.clip(np.asarray(r.x, dtype=float).reshape(-1), -5.0, 5.0)
                except Exception:
                    pass
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Phase C: second CMA-ES squeeze with tighter sigma.
        # ------------------------------------------------------------------
        if budget_left > 20 and time_ok():
            try:
                from cma import CMA
                res = CMA(evolve=wrap, x0=best_x.tolist(),
                          sigma0=0.2,
                          bounds=[(-5.0,) * dim, (5.0,) * dim],
                          maxfevals=max(15, budget_left),
                          seed=2,
                          verbose=-9, quiet=True,
                          pickle=False,
                          toff=0, cmaes_tolerances={
                              'tolask'  : 0.0,
                              'tolfun'  : 1e-15,
                              'tolx'    : 1e-12,
                              'tolopt'  : 0.0,
                          })
                if res.fbest <= best_f:
                    best_f = float(res.fbest)
                    best_x = np.array(res.xbest, dtype=float).reshape(dim)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Phase D: Final high-precision polish with BFGS (more reliable than NM for smooth functions)
        # ------------------------------------------------------------------
        if budget_left > 5 and time_ok():
            try:
                from scipy.optimize import minimize
                r = minimize(wrap, best_x, method='BFGS',
                             bounds=[(-5.0, 5.0)] * dim,
                             options={'maxiter': 100, 'gtol': 1e-14, 'disp': False})
                budget_left -= int(r.nfev)
                if r.fun <= best_f:
                    best_f = float(r.fun)
                    best_x = np.clip(np.asarray(r.x, dtype=float).reshape(-1), -5.0, 5.0)
            except Exception:
                pass

        return best_x, float(best_f)
# EVOLVE-BLOCK-END
