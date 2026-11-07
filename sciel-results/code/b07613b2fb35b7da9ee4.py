# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using structured exploration and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_mask(self, F):
        """Return a boolean mask of non-dominated points in F (n x n_obj array)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)

        F_i = F[:, np.newaxis, :]  # (n, 1, n_obj)
        F_j = F[np.newaxis, :, :]  # (1, n, n_obj)

        leq = np.all(F_i <= F_j, axis=2)  # (n, n)
        lt = np.any(F_i < F_j, axis=2)    # (n, n)

        dominates = leq & lt
        dominated = np.any(dominates, axis=0)

        return ~dominated

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(123)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        ref_point = np.array([1705.0, 12.0, 0.3])

        def time_left():
            return deadline - time.time()

        def is_deadline():
            return time.time() > deadline - 10

        # Phase 1: Structured initialization with LHS for better coverage
        n_initial = min(60, budget // 3)
        X = np.random.uniform(lo, hi, size=(n_initial, dim))
        
        # Add some structured points at corners and midpoints
        structured = []
        for i in range(2**dim):
            x = np.array([(hi[j] if (i >> j) & 1 else lo[j]) for j in range(dim)])
            structured.append(x)
        # Add midpoint
        structured.append((lo + hi) / 2)
        structured = np.array(structured)
        
        # Combine and sample
        all_init = np.vstack([X, structured])
        if len(all_init) > n_initial:
            idx = rng.choice(len(all_init), size=n_initial, replace=False)
            X = all_init[idx]
        else:
            X = all_init
        
        F = np.array([problem(x) for x in X])
        used = len(X)
        all_F = F.copy()

        # Phase 2: Evolutionary search with adaptive mutation
        pop_size = min(40, budget // 4)
        search_budget = int(budget * 0.65)

        while used < search_budget:
            if is_deadline():
                break

            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]

            if len(nd_idx) == 0:
                nd_idx = np.array([np.argmin(F[:, 0]), np.argmin(F[:, 1]), np.argmin(F[:, 2])])
                nd_idx = np.unique(nd_idx)

            progress = used / search_budget
            mut_scale = 0.35 * (0.990 ** used)
            mut_scale = max(mut_scale, 0.02)

            n_children = min(25, search_budget - used)
            for _ in range(n_children):
                if is_deadline():
                    break
                if used >= search_budget:
                    break

                # Parent selection: 80% from non-dominated, 20% random
                if rng.random() < 0.80:
                    parent_idx = nd_idx[rng.integers(len(nd_idx))]
                else:
                    parent_idx = rng.integers(len(F))

                parent = X[parent_idx].copy()

                # Mutation: Gaussian with occasional larger jumps
                if rng.random() < 0.10:
                    jump_scale = mut_scale * 5.0
                else:
                    jump_scale = mut_scale

                child = parent + jump_scale * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                all_F = np.vstack([all_F, fc])

                # Replacement strategy: replace dominated point or worst by hypervolume contribution
                worst_idx = self._find_worst_index(F, ref_point)
                
                # Check if child is better than worst
                child_dominates_worst = np.all(fc <= F[worst_idx]) and np.any(fc < F[worst_idx])
                worst_dominates_child = np.all(F[worst_idx] <= fc) and np.any(F[worst_idx] < fc)
                
                if child_dominates_worst or not worst_dominates_child:
                    X[worst_idx] = child
                    F[worst_idx] = fc

        # Phase 3: Local refinement with weighted scalarization (NSGA-II style)
        if used < budget and time_left() > 60:
            from scipy.optimize import minimize

            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]

            if len(nd_idx) > 0:
                # Generate more weight vectors for better coverage
                n_weights = min(20, max(5, (budget - used) // 4))
                W = rng.random((n_weights, n_obj))
                W /= W.sum(axis=1, keepdims=True)

                # Sort non-dominated points by distance from origin (prioritize extreme points)
                F_nd = F[nd_idx]
                F_min = F_nd.min(axis=0)
                F_max = F_nd.max(axis=0)
                F_range = F_max - F_min
                F_range[F_range == 0] = 1.0
                F_norm = (F_nd - F_min) / F_range
                dists = np.sqrt(np.sum(F_norm ** 2, axis=1))
                sorted_nd = nd_idx[np.argsort(dists)]

                points_to_refine = sorted_nd[:min(12, len(sorted_nd))]
                for pt_idx in points_to_refine:
                    if used >= budget or time_left() < 60:
                        break

                    x0 = X[pt_idx].copy()

                    # Use multiple weight vectors for each point
                    for w in W[:6]:
                        if used >= budget or time_left() < 60:
                            break

                        eval_count = [0]

                        def obj(x, w=w):
                            eval_count[0] += 1
                            return np.dot(w, problem(x))

                        try:
                            res = minimize(obj, x0, method='L-BFGS-B',
                                          bounds=list(zip(lo, hi)),
                                          options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
                            if res.success and eval_count[0] > 0:
                                fx = problem(res.x)
                                used += eval_count[0] + 1
                                all_F = np.vstack([all_F, fx])

                                # Update if improved
                                if np.any(fx < F[pt_idx]):
                                    X[pt_idx] = res.x
                                    F[pt_idx] = fx
                        except:
                            used += eval_count[0]

        return X

    def _find_worst_index(self, F, ref_point):
        """Find the index of the worst point in the population using hypervolume contribution."""
        if len(F) == 0:
            return 0

        nd_mask = self._non_dominated_mask(F)
        n = len(F)

        # Compute approximate hypervolume contribution for each point
        # Points that dominate others or are far from reference are "worse"
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1.0
        F_norm = (F - F_min) / F_range

        dist = np.sqrt(np.sum(F_norm ** 2, axis=1))
        penalty = np.where(~nd_mask, 100.0, 0.0)

        # Also penalize points that are close to the reference point (less useful)
        ref_dist = np.sqrt(np.sum((F - ref_point) ** 2, axis=1))
        ref_penalty = -0.1 * ref_dist  # Points closer to ref get higher penalty (worse)

        quality = dist + penalty + ref_penalty
        return np.argmax(quality)
# EVOLVE-BLOCK-END
