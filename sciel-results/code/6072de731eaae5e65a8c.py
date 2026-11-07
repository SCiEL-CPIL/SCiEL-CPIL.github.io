# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer for RE34 using evolutionary search + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_mask(self, F):
        """Return a boolean mask of non-dominated points in F (n x n_obj array)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        if n == 1:
            return np.array([True])

        F_i = F[:, np.newaxis, :]  # (n, 1, n_obj)
        F_j = F[np.newaxis, :, :]  # (1, n, n_obj)

        leq = np.all(F_i <= F_j, axis=2)
        lt = np.any(F_i < F_j, axis=2)

        dominates = leq & lt
        dominated = np.any(dominates, axis=0)
        return ~dominated

    def _hv_contribution(self, F, ref):
        """Compute hypervolume contribution of each point (approximate via sorting)."""
        n = len(F)
        if n == 0:
            return np.array([])
        if n == 1:
            d = np.maximum(ref - F[0], 0)
            return np.array([np.prod(d)])

        # Simple O(n^2) contribution: for each point, compute the volume it uniquely covers
        # This is approximate but good enough for selection
        contrib = np.zeros(n)
        for i in range(n):
            # Volume of box from F[i] to ref
            box = np.maximum(ref - F[i], 0)
            vol_i = np.prod(box)
            # Subtract volumes dominated by other points
            # A point j dominates i if F[j] <= F[i] and F[j] != F[i]
            # The overlap volume is the box from F[i] to ref minus parts covered by dominators
            # Approximation: for each other point j, if j is "better" in all dims, reduce contribution
            for j in range(n):
                if j == i:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    # j dominates i, so i's contribution is reduced
                    # The overlapping box is from F[i] to ref, but j already covers part
                    # Actually if j dominates i, then i's unique contribution is 0 in the region j covers
                    # Simplified: just mark as low contribution
                    vol_i = 0
                    break
            contrib[i] = vol_i
        return contrib

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        ref_point = np.array([1705.0, 12.0, 0.3])

        def time_left():
            return deadline - time.time()

        def is_deadline():
            return time.time() > deadline - 15

        # Phase 1: Structured initialization
        n_initial = min(50, budget // 4)
        
        # Latin Hypercube Sampling for better coverage
        def lhs_sample(n, dim, lo, hi, rng):
            X = np.zeros((n, dim))
            for d in range(dim):
                perm = rng.permutation(n)
                X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + rng.random(n)) / n
            return X

        X_lhs = lhs_sample(n_initial, dim, lo, hi, rng)
        
        # Add corner points and center
        structured = []
        for i in range(2**dim):
            x = np.array([(hi[j] if (i >> j) & 1 else lo[j]) for j in range(dim)])
            structured.append(x)
        structured.append((lo + hi) / 2)
        structured = np.array(structured)
        
        # Combine: use LHS points, replace some with structured
        n_structured = min(len(structured), n_initial // 3)
        if n_structured > 0:
            idx_replace = rng.choice(n_initial, size=n_structured, replace=False)
            X_init = X_lhs.copy()
            X_init[idx_replace] = structured[:n_structured]
        else:
            X_init = X_lhs

        F = np.array([problem(x) for x in X_init])
        used = len(X_init)
        all_F = F.copy()

        # Phase 2: Evolutionary search with adaptive mutation
        pop_size = len(X_init)
        search_budget = int(budget * 0.70)

        while used < search_budget:
            if is_deadline():
                break

            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]

            if len(nd_idx) == 0:
                nd_idx = np.array([np.argmin(F[:, 0]), np.argmin(F[:, 1]), np.argmin(F[:, 2])])
                nd_idx = np.unique(nd_idx)

            progress = used / max(search_budget, 1)
            # Adaptive mutation: start large, shrink over time
            mut_scale = 0.30 * (0.995 ** used)
            mut_scale = max(mut_scale, 0.015)

            n_children = min(20, search_budget - used)
            for _ in range(n_children):
                if is_deadline():
                    break
                if used >= search_budget:
                    break

                # Parent selection: tournament from non-dominated set
                if len(nd_idx) > 1 and rng.random() < 0.85:
                    # Tournament selection among non-dominated
                    t_size = min(3, len(nd_idx))
                    candidates = nd_idx[rng.choice(len(nd_idx), size=t_size, replace=False)]
                    # Pick the one with best HV contribution (approx: closest to ideal)
                    F_cand = F[candidates]
                    # Use distance to minimum as proxy
                    ideal = F_cand.min(axis=0)
                    dists = np.sqrt(np.sum((F_cand - ideal) ** 2, axis=1))
                    parent_idx = candidates[np.argmin(dists)]
                else:
                    parent_idx = rng.integers(len(F))

                parent = X[parent_idx].copy()

                # Mutation: Gaussian with occasional larger jumps
                if rng.random() < 0.08:
                    jump_scale = mut_scale * 4.0
                else:
                    jump_scale = mut_scale

                child = parent + jump_scale * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                all_F = np.vstack([all_F, fc])

                # Replacement: find worst point in population
                # Use simple criterion: dominated points are worse, then by distance to reference
                nd_mask_pop = self._non_dominated_mask(F)
                dominated_idx = np.where(~nd_mask_pop)[0]
                
                if len(dominated_idx) > 0:
                    # Replace a random dominated point
                    worst_idx = dominated_idx[rng.integers(len(dominated_idx))]
                else:
                    # All non-dominated: replace the one with worst HV contribution
                    # Approximate: the one furthest from the "ideal" point
                    ideal = F.min(axis=0)
                    dists = np.sqrt(np.sum((F - ideal) ** 2, axis=1))
                    worst_idx = np.argmax(dists)

                # Check if child should replace worst
                child_dominates_worst = np.all(fc <= F[worst_idx]) and np.any(fc < F[worst_idx])
                worst_dominates_child = np.all(F[worst_idx] <= fc) and np.any(F[worst_idx] < fc)
                
                if child_dominates_worst or not worst_dominates_child:
                    # Additional check: if neither dominates, use HV contribution
                    if not child_dominates_worst and not worst_dominates_child:
                        # Compare approximate HV contributions
                        # Simple heuristic: child is better if it's closer to the ideal in at least one dim
                        # and not worse in others by too much
                        improvement = np.sum(np.minimum(fc, F[worst_idx])) - np.sum(F[worst_idx])
                        if improvement < 0:  # child improves at least one objective
                            X[worst_idx] = child
                            F[worst_idx] = fc
                    else:
                        X[worst_idx] = child
                        F[worst_idx] = fc

        # Phase 3: Local refinement with weighted scalarization
        if used < budget and time_left() > 30:
            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]

            if len(nd_idx) > 0:
                # Generate weight vectors for better coverage
                n_weights = min(15, max(4, (budget - used) // 5))
                W = rng.random((n_weights, n_obj))
                W /= W.sum(axis=1, keepdims=True)

                # Sort non-dominated points by diversity (spread)
                F_nd = F[nd_idx]
                F_min = F_nd.min(axis=0)
                F_max = F_nd.max(axis=0)
                F_range = F_max - F_min
                F_range[F_range == 0] = 1.0
                F_norm = (F_nd - F_min) / F_range
                dists = np.sqrt(np.sum(F_norm ** 2, axis=1))
                sorted_nd = nd_idx[np.argsort(dists)]

                points_to_refine = sorted_nd[:min(10, len(sorted_nd))]
                for pt_idx in points_to_refine:
                    if used >= budget or time_left() < 30:
                        break

                    x0 = X[pt_idx].copy()

                    # Use a few weight vectors for each point
                    for w in W[:4]:
                        if used >= budget or time_left() < 30:
                            break

                        eval_count = [0]

                        def obj(x, w=w):
                            eval_count[0] += 1
                            return np.dot(w, problem(x))

                        try:
                            res = minimize(obj, x0, method='L-BFGS-B',
                                          bounds=list(zip(lo, hi)),
                                          options={'maxiter': 40, 'ftol': 1e-14, 'gtol': 1e-12})
                            if res.success and eval_count[0] > 0:
                                fx = problem(res.x)
                                used += eval_count[0] + 1
                                all_F = np.vstack([all_F, fx])

                                # Update if improved
                                if np.any(fx < F[pt_idx]):
                                    X[pt_idx] = res.x
                                    F[pt_idx] = fx
                        except Exception:
                            used += eval_count[0]

        return X
# EVOLVE-BLOCK-END
