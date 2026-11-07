"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Reference-point-based adaptive Tchebycheff multi-start L-BFGS-B."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        bounds_list = list(zip(lo, hi))
        n_obj = self.n_obj
        dim = self.dim

        archive = []  # list of (x, f)
        used = 0

        def time_left():
            return time.time() < deadline

        def add_point(x, f):
            nonlocal used
            if used >= self.budget:
                return False
            if x is None or f is None:
                return False
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.asarray(f, dtype=float)
            if not np.all(np.isfinite(f)):
                return False
            archive.append((x, f))
            used += 1
            return True

        def is_dominated(f, archive_f):
            """Check if f is dominated by any point in archive_f (list of arrays)."""
            for af in archive_f:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def non_dominated_front(points_f):
            """Return indices of non-dominated points."""
            n = len(points_f)
            if n == 0:
                return []
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    fi = points_f[i]
                    fj = points_f[j]
                    if np.all(fi <= fj) and np.any(fi < fj):
                        dominated[j] = True
            return [i for i in range(n) if not dominated[i]]

        # Phase 1: Initial diverse sampling
        n_init = 40
        X_init = np.empty((n_init, dim))
        for d in range(dim):
            X_init[:, d] = lo[d] + (hi[d] - lo[d]) * (np.arange(n_init) + rng.random()) / n_init
        rng.shuffle(X_init)

        for x in X_init:
            if used >= self.budget or not time_left():
                break
            f = problem(x)
            add_point(x, f)

        # Phase 2: Adaptive Tchebycheff decomposition
        # We'll iteratively choose weight vectors that target gaps in the current front
        n_rounds = 15  # number of Tchebycheff sub-problems to solve
        max_starts_per_round = 4
        maxfun_per_min = 30

        # Precompute reference point for Tchebycheff: use nadir approximation
        def get_nadir():
            if not archive:
                return np.ones(n_obj) * 10.0
            f_arr = np.array([f for _, f in archive])
            return f_arr.max(axis=0)

        def tchebycheff_obj(x, weights, z_ideal, z_nadir):
            f = problem(x)
            # Tchebycheff distance
            dists = np.abs(weights * (f - z_ideal)) / (z_nadir - z_ideal + 1e-10)
            return float(np.max(dists))

        # Generate initial weight vectors on simplex
        def gen_weights(n):
            weights = []
            # Uniform on simplex via Dirichlet
            for _ in range(n):
                w = rng.dirichlet(np.ones(n_obj))
                weights.append(w)
            # Add extreme points
            for i in range(n_obj):
                w = np.zeros(n_obj)
                w[i] = 1.0
                weights.append(w)
            # Add mid-range
            for i in range(n_obj):
                for j in range(i + 1, n_obj):
                    w = np.zeros(n_obj)
                    w[i] = 0.5
                    w[j] = 0.5
                    weights.append(w)
            return weights

        all_weights = gen_weights(n_rounds)
        rng.shuffle(all_weights)

        z_ideal = np.ones(n_obj) * np.inf
        z_nadir = np.zeros(n_obj)

        def update_ideals():
            nonlocal z_ideal, z_nadir
            if not archive:
                return
            f_arr = np.array([f for _, f in archive])
            z_ideal = f_arr.min(axis=0)
            z_nadir = f_arr.max(axis=0)

        update_ideals()

        # For each weight, do multi-start L-BFGS-B
        for w in all_weights:
            if used >= self.budget or not time_left():
                break

            update_ideals()
            zn = z_nadir - z_ideal + 1e-10
            zn = np.where(zn < 1e-8, 1.0, zn)

            def obj_func(x, w=w, zn=zn, zi=z_ideal):
                f = problem(x)
                if not np.all(np.isfinite(f)):
                    return 1e10
                dists = np.abs(w * (f - zi)) / zn
                return float(np.max(dists))

            # Starting points: mix of random, archive points, and boundary points
            starts = []
            # Random starts
            for _ in range(2):
                starts.append(lo + (hi - lo) * rng.random(dim))
            # Archive starts
            if archive:
                n_arch = len(archive)
                idxs = rng.choice(n_arch, size=min(3, n_arch), replace=False)
                for idx in idxs:
                    starts.append(archive[idx][0].copy())
            # Boundary points: push one variable to its bound
            for d in range(dim):
                xb = lo + (hi - lo) * rng.random(dim)
                xb[d] = hi[d] if rng.random() > 0.5 else lo[d]
                starts.append(xb)

            best_fun = np.inf
            best_x = None
            best_f = None

            for x0 in starts:
                if used >= self.budget or not time_left():
                    break
                try:
                    res = minimize(obj_func, x0, method='L-BFGS-B',
                                   bounds=bounds_list,
                                   options={'maxiter': 150, 'maxfun': maxfun_per_min})
                    if res.fun < best_fun and np.isfinite(res.fun):
                        x_opt = np.clip(res.x, lo, hi)
                        f_opt = problem(x_opt)
                        if np.all(np.isfinite(f_opt)):
                            best_fun = res.fun
                            best_x = x_opt
                            best_f = f_opt
                except Exception:
                    pass

            if best_x is not None:
                add_point(best_x, best_f)

        # Phase 3: Local refinement around current non-dominated front
        if archive and time_left():
            f_arr = np.array([f for _, f in archive])
            nd_idx = non_dominated_front(f_arr)
            if nd_idx:
                # For each non-dominated point, do a small local search
                for idx in nd_idx:
                    if used >= self.budget or not time_left():
                        break
                    x0 = archive[idx][0]
                    try:
                        # Use a small perturbation-based local search
                        for _ in range(3):
                            if used >= self.budget or not time_left():
                                break
                            x_pert = x0 + (hi - lo) * 0.05 * rng.standard_normal(dim)
                            x_pert = np.clip(x_pert, lo, hi)
                            f_pert = problem(x_pert)
                            if np.all(np.isfinite(f_pert)):
                                add_point(x_pert, f_pert)
                    except Exception:
                        pass

        return archive
# EVOLVE-BLOCK-END
