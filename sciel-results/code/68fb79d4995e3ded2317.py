# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model-and-optimize multi-objective optimizer for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, pts, ref):
        """Exact hypervolume for 2D points in O(n log n)."""
        n = len(pts)
        if n == 0:
            return 0.0
        if n == 1:
            return float((ref[0] - pts[0, 0]) * (ref[1] - pts[0, 1]))
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        hv = 0.0
        y_min = ref[1]
        for i in range(n - 1, -1, -1):
            if pts[i, 1] < y_min:
                y_min = pts[i, 1]
                x_cur = pts[i, 0]
                x_next = pts[i + 1, 0] if i + 1 < n else ref[0]
                hv += (x_next - x_cur) * (ref[1] - y_min)
        return hv

    def _hv_3d(self, pts, ref):
        """Exact hypervolume for 3D points using sweep in O(n^2)."""
        n = len(pts)
        if n == 0:
            return 0.0
        mask = np.all(pts < ref, axis=1)
        pts = pts[mask]
        n = len(pts)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(ref - pts[0]))
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        hv = 0.0
        for i in range(n):
            x_cur = pts[i, 0]
            x_next = pts[i + 1, 0] if i + 1 < n else ref[0]
            yz = pts[:i + 1, 1:3]
            hv += (x_next - x_cur) * self._hv_2d(yz, ref[1:3])
        return hv

    def _delta_hv(self, pts, ref, add_pt):
        """Approximate HV contribution of adding add_pt to pts."""
        # Quick check: is add_pt dominated by existing points?
        if len(pts) > 0:
            if np.any(np.all(pts <= add_pt, axis=1) & np.any(pts < add_pt, axis=1)):
                return 0.0
        # Compute HV with and without the point
        pts_all = np.vstack([pts, [add_pt]]) if len(pts) > 0 else add_pt.reshape(1, -1)
        hv_with = self._hv_3d(pts_all, ref)
        hv_without = self._hv_3d(pts, ref) if len(pts) > 0 else 0.0
        return hv_with - hv_without

    def _is_pareto(self, F):
        """Return boolean mask of non-dominated points."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        pareto = np.ones(n, dtype=bool)
        for i in range(n):
            if not pareto[i]:
                continue
            others = np.delete(F, i, axis=0)
            if len(others) == 0:
                continue
            dominated = np.any(np.all(others <= F[i], axis=1) & np.any(others < F[i], axis=1))
            if dominated:
                pareto[i] = False
        return pareto

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1705.0, 12.0, 0.3])

        # --- Phase 1: Initial sampling ---
        n_lhs = 35
        n_rand = 15
        n_init = min(n_lhs + n_rand, budget)

        lhs_X = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = np.random.permutation(n_lhs)
            lhs_X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + np.random.uniform(size=n_lhs)) / n_lhs
        lhs_X = np.clip(lhs_X, lo, hi)

        rand_X = lo + np.random.uniform(size=(n_rand, dim)) * (hi - lo)
        rand_X = np.clip(rand_X, lo, hi)

        X_init = np.vstack([lhs_X, rand_X])
        F_init = np.array([problem(x) for x in X_init])
        used = len(X_init)

        all_X = X_init.copy()
        all_F = F_init.copy()

        # Build initial Pareto archive
        valid = np.all(F_init < ref, axis=1)
        if np.any(valid):
            pF = F_init[valid]
            pX = X_init[valid]
            p_mask = self._is_pareto(pF)
            pareto_X = pX[p_mask].copy()
            pareto_F = pF[p_mask].copy()
        else:
            pareto_X = np.empty((0, dim))
            pareto_F = np.empty((0, n_obj))

        # --- Phase 2: Main search loop ---
        polish_start = int(budget * 0.70)

        while used < budget:
            if time.time() > deadline:
                break

            progress = used / budget
            is_polish = used >= polish_start

            # Parent selection
            if len(pareto_F) > 0:
                valid_pareto = np.all(pareto_F < ref, axis=1)
                if np.any(valid_pareto):
                    pX = pareto_X[valid_pareto]
                    pF = pareto_F[valid_pareto]
                else:
                    pX = pareto_X
                    pF = pareto_F

                # Use delta-HV based selection for better guidance
                # Compute approximate contributions for a subset to save time
                if len(pF) <= 20:
                    contribs = np.zeros(len(pF))
                    for i in range(len(pF)):
                        others = np.delete(pF, i, axis=0)
                        contribs[i] = self._delta_hv(others, ref, pF[i])
                else:
                    # For larger archives, use distance-based proxy
                    dists = np.sum(np.abs(pF - ref), axis=1)
                    contribs = 1.0 / (dists + 1e-10)

                # Add diversity: prefer points that are well-spread
                # Use a combination of contribution and angular diversity
                if len(pF) > 1:
                    # Compute pairwise distances for diversity
                    from scipy.spatial.distance import cdist
                    dist_matrix = cdist(pF, pF)
                    avg_dist = dist_matrix.sum(axis=1) / max(1, len(pF) - 1)
                    # Normalize
                    contribs_norm = contribs / (contribs.max() + 1e-10)
                    avg_dist_norm = avg_dist / (avg_dist.max() + 1e-10)
                    # Combine
                    temp = 1.0 + 1.5 * (1.0 - progress)
                    weights = contribs_norm ** temp * (0.5 + 0.5 * avg_dist_norm)
                    weights /= (weights.sum() + 1e-10)
                else:
                    weights = np.ones(1)

                parent_idx = np.random.choice(len(pF), p=weights)
                parent = pX[parent_idx]
            else:
                if len(all_X) > 0:
                    parent = all_X[np.random.randint(len(all_X))]
                else:
                    parent = lo + np.random.uniform(size=dim) * (hi - lo)

            # Adaptive step size
            if is_polish:
                remaining_frac = (budget - used) / max(1, budget - polish_start)
                base_step = 0.015 + 0.06 * remaining_frac
            else:
                base_step = 0.45 * (1.0 - progress * 0.75) + 0.08

            step = base_step * (0.6 + 0.8 * np.random.uniform())

            # Coordinate-wise or full mutation
            coord_prob = 0.35 + 0.45 * progress
            use_coord = np.random.uniform() < coord_prob

            if not use_coord:
                child = parent + step * (hi - lo) * np.random.standard_normal(dim)
            else:
                child = parent.copy()
                n_perturb = np.random.randint(1, min(3, dim + 1))
                coords = np.random.choice(dim, size=n_perturb, replace=False)
                for c in coords:
                    child[c] = parent[c] + step * (hi[c] - lo[c]) * np.random.standard_normal()
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            all_X = np.vstack([all_X, [child]])
            all_F = np.vstack([all_F, [fc]])

            # Update Pareto archive
            if np.all(fc < ref):
                dominated = False
                for pf in pareto_F:
                    if np.all(pf <= fc) and np.any(pf < fc):
                        dominated = True
                        break

                if not dominated:
                    pareto_X = np.vstack([pareto_X, [child]])
                    pareto_F = np.vstack([pareto_F, [fc]])
                    keep = np.ones(len(pareto_F), dtype=bool)
                    for i in range(len(pareto_F) - 1):
                        if np.all(pareto_F[i] >= fc) and np.any(pareto_F[i] > fc):
                            keep[i] = False
                    pareto_X = pareto_X[keep]
                    pareto_F = pareto_F[keep]

                # Prune archive if too large
                if len(pareto_F) > 35:
                    p_mask = self._is_pareto(pareto_F)
                    pareto_X = pareto_X[p_mask]
                    pareto_F = pareto_F[p_mask]
                    if len(pareto_F) > 35:
                        valid_prune = np.all(pareto_F < ref, axis=1)
                        if np.any(valid_prune):
                            pf_valid = pareto_F[valid_prune]
                            px_valid = pareto_X[valid_prune]
                            # Keep points with best spread
                            dists = np.sum(np.abs(pf_valid - ref), axis=1)
                            idx_sorted = np.argsort(dists)
                            pareto_X = px_valid[idx_sorted[:35]]
                            pareto_F = pf_valid[idx_sorted[:35]]
                        else:
                            keep_idx = np.random.choice(len(pareto_X), size=35, replace=False)
                            pareto_X = pareto_X[keep_idx]
                            pareto_F = pareto_F[keep_idx]

        # --- Phase 3: Local refinement ---
        if used < budget and len(pareto_F) > 0:
            valid_final = np.all(pareto_F < ref, axis=1)
            if np.any(valid_final):
                pX_final = pareto_X[valid_final]
                pF_final = pareto_F[valid_final]
            else:
                pX_final = pareto_X
                pF_final = pareto_F

            if len(pX_final) > 0:
                # Refine top points by sum of objectives
                sums = np.sum(pF_final, axis=1)
                top_k = min(6, len(pX_final))
                top_idx = np.argsort(sums)[:top_k]

                for pi in top_idx:
                    if used >= budget or time.time() > deadline - 5:
                        break
                    parent = pX_final[pi].copy()

                    # Multi-directional local search
                    for coord in range(dim):
                        if used >= budget or time.time() > deadline - 3:
                            break

                        # Try a range of small perturbations
                        best_f = problem(parent).copy()
                        best_x = parent.copy()
                        span = hi[coord] - lo[coord]

                        for delta_frac in [-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08]:
                            if used >= budget or time.time() > deadline - 2:
                                break
                            trial = parent.copy()
                            trial[coord] = np.clip(parent[coord] + delta_frac * span, lo[coord], hi[coord])
                            if np.allclose(trial, parent):
                                continue
                            ft = problem(trial)
                            used += 1
                            if np.all(ft <= best_f) and np.any(ft < best_f):
                                best_f = ft
                                best_x = trial

                        if not np.allclose(best_x, parent):
                            parent = best_x
                            all_X = np.vstack([all_X, [parent]])
                            all_F = np.vstack([all_F, [best_f]])

                            if np.all(best_f < ref):
                                dominated = False
                                for pf in pareto_F:
                                    if np.all(pf <= best_f) and np.any(pf < best_f):
                                        dominated = True
                                        break
                                if not dominated:
                                    pareto_X = np.vstack([pareto_X, [parent]])
                                    pareto_F = np.vstack([pareto_F, [best_f]])
                                    keep = np.ones(len(pareto_F), dtype=bool)
                                    for i in range(len(pareto_F) - 1):
                                        if np.all(pareto_F[i] >= best_f) and np.any(pareto_F[i] > best_f):
                                            keep[i] = False
                                    pareto_X = pareto_X[keep]
                                    pareto_F = pareto_F[keep]

        return all_X

# EVOLVE-BLOCK-END
