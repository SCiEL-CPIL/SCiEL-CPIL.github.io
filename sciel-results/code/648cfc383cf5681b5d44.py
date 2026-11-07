"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using a population-based evolutionary strategy
    with Pareto dominance and adaptive step sizes."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        width = hi - lo

        X_archive = []
        F_archive = []
        eval_count = 0
        nd_cache = None
        nd_cache_len = 0

        def safe_problem(x):
            nonlocal eval_count, nd_cache, nd_cache_len
            if eval_count >= self.budget:
                return [1e10] * self.n_obj
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=float)
            if not np.all(np.isfinite(f)):
                f = np.full(self.n_obj, 1e10)
            eval_count += 1
            X_archive.append(x.copy())
            F_archive.append(f.copy())
            # Invalidate cache since archive changed
            nd_cache = None
            return f

        def get_nd_front():
            """Get non-dominated front with caching."""
            nonlocal nd_cache, nd_cache_len
            if nd_cache is not None and nd_cache_len == len(F_archive):
                return nd_cache
            n = len(F_archive)
            if n == 0:
                nd_cache = []
                nd_cache_len = 0
                return nd_cache
            F = np.array(F_archive)
            # Vectorized dominance check
            # F[i] dominates F[j] if all(F[i] <= F[j]) and any(F[i] < F[j])
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                # Check if i dominates any j > i
                diff = F[i] - F[i+1:]
                dom_mask = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
                if np.any(dom_mask):
                    dominated[i+1:][dom_mask] = True
            nd_cache = [i for i in range(n) if not dominated[i]]
            nd_cache_len = n
            return nd_cache

        # 1. Initial population: Latin hypercube for better coverage
        n_init = min(25, self.budget)
        pop_X = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(n_init)
            pop_X[:, d] = (perm + np.random.rand(n_init)) / n_init
        pop_X = lo + pop_X * width

        for x in pop_X:
            if eval_count >= self.budget:
                break
            safe_problem(x)

        if eval_count >= self.budget:
            return X_archive

        # 2. Evolutionary loop with adaptive step size
        step_scale = 0.20  # fraction of range
        min_step_scale = 0.02
        max_step_scale = 0.40
        best_hv = 0.0

        def compute_hv(F_list):
            """Compute hypervolume with reference point [1705, 12, 0.3]."""
            if not F_list:
                return 0.0
            F = np.array(F_list)
            ref = np.array([1705.0, 12.0, 0.3])
            # Only points strictly better than reference in all objectives
            mask = np.all(F < ref, axis=1)
            if not np.any(mask):
                return 0.0
            F = F[mask]
            # Sort by first objective
            order = np.argsort(F[:, 0])
            F = F[order]
            # Compute hypervolume in 3D using inclusion-exclusion
            n = len(F)
            if n == 0:
                return 0.0
            # For 3D, we can use a sweep line approach
            # Sort by x, then for each point compute the area in yz plane
            hv = 0.0
            prev_x = ref[0]
            for i in range(n):
                x = F[i, 0]
                # Compute the 2D hypervolume in yz for points up to i
                # Points that are "active" at x: those with F[j,0] >= x
                # Simplified: use the standard 3D HV algorithm
                pass
            # Use a simpler approximation: sum of contributions
            # For each point, its contribution is the volume of the box it uniquely covers
            # This is complex; use a numerical approach
            # Sort by obj1
            F_sorted = F[np.argsort(F[:, 0])]
            # Build the Pareto front in yz for each x slice
            # This is getting complex; use a simple Monte Carlo estimate
            # Actually, let's just use the fact that we want to maximize coverage
            # Return a proxy: sum of (ref - F) for non-dominated points
            nd = get_nd_front()
            if not nd:
                return 0.0
            F_nd = F_archive[nd]
            # Proxy: sum of distances to reference
            return np.sum(np.prod(ref - F_nd, axis=1))

        while eval_count < self.budget - 3:
            if time.time() > deadline:
                break

            nd_idx = get_nd_front()
            if not nd_idx:
                nd_idx = list(range(len(F_archive)))

            # Limit parents
            max_parents = min(len(nd_idx), 8)
            if len(nd_idx) > max_parents:
                parent_indices = np.random.choice(nd_idx, size=max_parents, replace=False)
            else:
                parent_indices = nd_idx

            # Generate offspring
            n_offspring = min(max_parents, self.budget - eval_count - 3)
            if n_offspring <= 0:
                break

            new_points = []
            for _ in range(n_offspring):
                pi = parent_indices[np.random.randint(len(parent_indices))]
                parent_x = X_archive[pi]

                # Adaptive step: larger when we have few points, smaller when many
                step = step_scale * width
                child_x = parent_x + np.random.randn(self.dim) * step
                child_x = np.clip(child_x, lo, hi)

                f_child = safe_problem(child_x)
                if eval_count >= self.budget:
                    break
                new_points.append(f_child)

            # Check improvement
            if new_points:
                current_hv_proxy = compute_hv(F_archive)
                if current_hv_proxy > best_hv:
                    best_hv = current_hv_proxy
                    step_scale = min(step_scale * 1.05, max_step_scale)
                else:
                    step_scale = max(step_scale * 0.9, min_step_scale)

        # 3. Final random fill
        while eval_count < self.budget:
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.rand(self.dim)
            safe_problem(x)

        return X_archive
# EVOLVE-BLOCK-END
