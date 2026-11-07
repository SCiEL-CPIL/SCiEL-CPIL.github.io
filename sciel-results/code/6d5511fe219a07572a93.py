"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer with Tchebycheff-based adaptive scalarization and L-BFGS-B."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, pts, ref_2):
        if len(pts) == 0:
            return 0.0
        msk = np.all(pts < ref_2, axis=1)
        p = pts[msk]
        if len(p) == 0:
            return 0.0
        p = p[np.argsort(p[:, 0])]
        n_p = len(p)
        hv = 0.0
        for i in range(n_p):
            next_x = p[i + 1, 0] if i + 1 < n_p else ref_2[0]
            width = next_x - p[i, 0]
            height = ref_2[1] - p[i, 1]
            hv += width * height
        return hv

    def _hv_3d(self, pts, ref):
        if len(pts) == 0:
            return 0.0
        msk = np.all(pts < ref, axis=1)
        p = pts[msk]
        if len(p) == 0:
            return 0.0
        p = p[np.argsort(p[:, 0])]
        n_p = len(p)
        hv = 0.0
        for i in range(n_p):
            next_f0 = p[i + 1, 0] if i + 1 < n_p else ref[0]
            width = next_f0 - p[i, 0]
            sub_pts = p[i:, 1:]
            area = self._hv_2d(sub_pts, ref[1:])
            hv += width * area
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])

        used = 0
        solutions = []

        def eval_point(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            used += 1
            return x, f

        def time_check():
            return time.time() > deadline

        # Phase 1: Latin Hypercube Sampling
        n_lhs = 30
        X_lhs = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(n_lhs)
            X_lhs[:, d] = (perm + rng.random(n_lhs)) / n_lhs
        X_lhs = lo + X_lhs * (hi - lo)

        for x in X_lhs:
            if used >= self.budget or time_check():
                break
            x_c, f = eval_point(x)
            solutions.append((x_c, f))

        # Phase 2: Adaptive Tchebycheff scalarization with L-BFGS-B
        # We'll maintain an archive and use Tchebycheff to fill gaps

        def get_pareto_front(sols):
            if len(sols) == 0:
                return np.array([]), np.array([])
            X = np.array([x for x, _ in sols])
            F = np.array([f for _, f in sols])
            n = len(X)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
            nd_idx = np.where(~dominated)[0]
            return X[nd_idx], F[nd_idx]

        def tchebycheff_objective(x, w, ideal, rho=0.05):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            used += 1
            return np.max(w * np.abs(f - ideal)) + rho * np.sum(w * (f - ideal))

        def run_tchebycheff(x0, w, ideal, rho=0.05):
            nonlocal used
            if used >= self.budget or time_check():
                return None
            bounds_scipy = list(zip(lo, hi))
            try:
                res = minimize(
                    tchebycheff_objective,
                    x0,
                    args=(w, ideal, rho),
                    method='L-BFGS-B',
                    bounds=bounds_scipy,
                    options={'maxiter': 50, 'maxfun': 30, 'gtol': 1e-6}
                )
                if time_check() or used >= self.budget:
                    return None
                x_res = np.clip(res.x, lo, hi)
                x_res, f_res = eval_point(x_res)
                return x_res, f_res
            except Exception:
                return None

        # Main loop: adaptive Tchebycheff search
        max_iters = 50
        for it in range(max_iters):
            if used >= self.budget or time_check():
                break

            # Get current Pareto front
            X_pf, F_pf = get_pareto_front(solutions)
            if len(X_pf) == 0:
                break

            # Compute ideal point from current front
            ideal = np.min(F_pf, axis=0)

            # Generate Tchebycheff weights to fill gaps
            # Use a set of weights that cover the simplex
            n_weights = 8
            weights = []
            for i in range(n_weights):
                # Generate diverse weights
                w = rng.dirichlet(np.ones(self.n_obj) * 2.0)
                weights.append(w)

            # For each weight, run Tchebycheff from multiple starting points
            for w in weights:
                if used >= self.budget or time_check():
                    break

                # Choose starting point: either from Pareto front or random
                if len(X_pf) > 0 and rng.random() < 0.7:
                    # Start from a Pareto point
                    idx = rng.integers(len(X_pf))
                    x0 = X_pf[idx].copy()
                    # Add small perturbation
                    x0 += 0.05 * (hi - lo) * rng.standard_normal(self.dim)
                    x0 = np.clip(x0, lo, hi)
                else:
                    # Random start
                    x0 = lo + rng.random(self.dim) * (hi - lo)

                result = run_tchebycheff(x0, w, ideal)
                if result is not None:
                    solutions.append(result)

        # Phase 3: Final polish of Pareto front
        if used < self.budget and not time_check():
            X_pf, F_pf = get_pareto_front(solutions)
            if len(X_pf) > 0:
                ideal = np.min(F_pf, axis=0)
                for idx in range(len(X_pf)):
                    if used >= self.budget or time_check():
                        break
                    x_center = X_pf[idx]
                    # Try a few different weight vectors
                    for _ in range(3):
                        if used >= self.budget or time_check():
                            break
                        w = rng.dirichlet(np.ones(self.n_obj) * 3.0)
                        result = run_tchebycheff(x_center, w, ideal, rho=0.03)
                        if result is not None:
                            solutions.append(result)

        # Return the last solution
        if len(solutions) > 0:
            return solutions[-1][0]
        return lo.copy()
# EVOLVE-BLOCK-END
