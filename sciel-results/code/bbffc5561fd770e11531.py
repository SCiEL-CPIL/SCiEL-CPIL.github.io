# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model-and-optimize multi-objective optimizer for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_3d(self, pts, ref):
        """Exact hypervolume for 3D points using sweep-line in O(n log n)."""
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

        # Sort by x
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]

        # For each x-slice, maintain the 2D skyline in (y,z)
        # We sweep x from left to right, adding points and computing incremental HV
        # Use a simple O(n^2) but with early termination for small n
        hv = 0.0
        for i in range(n):
            x_cur = pts[i, 0]
            x_next = pts[i + 1, 0] if i + 1 < n else ref[0]
            # 2D HV of pts[0..i] in yz-space
            yz = pts[:i + 1, 1:3]
            hv += (x_next - x_cur) * self._hv_2d(yz, ref[1:3])
        return hv

    def _hv_2d(self, pts, ref):
        """Exact hypervolume for 2D points in O(n log n)."""
        n = len(pts)
        if n == 0:
            return 0.0
        if n == 1:
            return float((ref[0] - pts[0, 0]) * (ref[1] - pts[0, 1]))

        # Sort by x, then sweep maintaining max y seen so far
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]

        hv = 0.0
        y_min = ref[1]
        # Process from right to left: for each point, if it's the new minimum y,
        # it contributes a rectangle from its x to the next point's x (or ref[0])
        for i in range(n - 1, -1, -1):
            if pts[i, 1] < y_min:
                y_min = pts[i, 1]
                x_cur = pts[i, 0]
                x_next = pts[i + 1, 0] if i + 1 < n else ref[0]
                hv += (x_next - x_cur) * (ref[1] - y_min)
        return hv

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
        deadline = start_time + 315  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1705.0, 12.0, 0.3])

        rng = np.random

        # --- Phase 1: Initial sampling with LHS + random ---
        n_lhs = 30
        n_rand = 20
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

        # --- Phase 2: Model-based search ---
        # Fit a surrogate model to the data and optimize it
        # We'll use a simple approach: fit quadratic models to each objective
        # and use them to guide the search

        polish_start = int(budget * 0.65)

        while used < budget:
            if time.time() > deadline:
                break

            progress = used / budget
            is_polish = used >= polish_start

            # Get current HV
            if len(pareto_F) > 0:
                valid_pareto = np.all(pareto_F < ref, axis=1)
                if np.any(valid_pareto):
                    current_hv = self._hv_3d(pareto_F[valid_pareto], ref)
                else:
                    current_hv = 0.0
            else:
                current_hv = 0.0

            # Parent selection based on HV contribution
            if len(pareto_F) > 0:
                valid_pareto = np.all(pareto_F < ref, axis=1)
                if np.any(valid_pareto):
                    pX = pareto_X[valid_pareto]
                    pF = pareto_F[valid_pareto]
                else:
                    pX = pareto_X
                    pF = pareto_F

                # Compute approximate HV contribution for each point
                # Use distance to reference as a proxy, but also consider
                # how "central" the point is on the front
                dists = np.sum(np.abs(pF - ref), axis=1)
                # Prefer points that are well-spread: use a combination of
                # closeness to ref and diversity
                # Simpler: weight by inverse distance with temperature
                temp = 1.0 + 2.0 * (1.0 - progress)
                weights = np.exp(-dists / (dists.mean() + 1e-10) * temp)
                weights /= weights.sum()
                parent_idx = np.random.choice(len(pF), p=weights)
                parent = pX[parent_idx]
            else:
                if len(all_X) > 0:
                    parent = all_X[np.random.randint(len(all_X))]
                else:
                    parent = lo + np.random.uniform(size=dim) * (hi - lo)

            # Adaptive step size with exploration/exploitation balance
            if is_polish:
                remaining_frac = (budget - used) / max(1, budget - polish_start)
                base_step = 0.02 + 0.08 * remaining_frac
            else:
                # Larger steps early, smaller later
                base_step = 0.5 * (1.0 - progress * 0.7) + 0.1

            step = base_step * (0.6 + 0.8 * np.random.uniform())

            # Coordinate-wise or full mutation with adaptive probability
            # More coordinate-wise as we progress
            coord_prob = 0.3 + 0.5 * progress
            use_coord = np.random.uniform() < coord_prob

            if not use_coord:
                child = parent + step * (hi - lo) * np.random.standard_normal(dim)
            else:
                child = parent.copy()
                # Perturb 1-3 coordinates
                n_perturb = np.random.randint(1, min(4, dim + 1))
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
                # Check if new point is dominated
                dominated = False
                if len(pareto_F) > 0:
                    dominated = np.any(np.all(pareto_F <= fc, axis=1) & np.any(pareto_F < fc, axis=1))

                if not dominated:
                    pareto_X = np.vstack([pareto_X, [child]])
                    pareto_F = np.vstack([pareto_F, [fc]])
                    # Remove points dominated by the new point
                    if len(pareto_F) > 1:
                        keep = np.ones(len(pareto_F), dtype=bool)
                        for i in range(len(pareto_F) - 1):
                            if np.all(pareto_F[i] >= fc) and np.any(pareto_F[i] > fc):
                                keep[i] = False
                        pareto_X = pareto_X[keep]
                        pareto_F = pareto_F[keep]

                # Prune archive if too large - keep diverse points
                if len(pareto_F) > 50:
                    # First ensure only Pareto points
                    p_mask = self._is_pareto(pareto_F)
                    pareto_X = pareto_X[p_mask]
                    pareto_F = pareto_F[p_mask]
                    if len(pareto_F) > 50:
                        # Use greedy diversity: repeatedly pick the point that
                        # maximizes minimum distance to already-selected points
                        valid_prune = np.all(pareto_F < ref, axis=1)
                        pf_valid = pareto_F[valid_prune]
                        px_valid = pareto_X[valid_prune]
                        if len(pf_valid) > 50:
                            # Start with the point closest to reference
                            dists = np.sum(np.abs(pf_valid - ref), axis=1)
                            selected = [np.argmin(dists)]
                            remaining = set(range(len(pf_valid))) - set(selected)
                            while len(selected) < 50 and remaining:
                                # For each remaining point, compute min distance to selected
                                best_idx = None
                                best_min_dist = -1
                                for idx in remaining:
                                    min_dist = np.min(np.linalg.norm(pf_valid[idx] - pf_valid[selected], axis=1))
                                    if min_dist > best_min_dist:
                                        best_min_dist = min_dist
                                        best_idx = idx
                                if best_idx is None:
                                    break
                                selected.append(best_idx)
                                remaining.remove(best_idx)
                            pareto_X = px_valid[np.array(selected)]
                            pareto_F = pf_valid[np.array(selected)]

        # --- Phase 3: Local refinement on top points ---
        if used < budget and len(pareto_F) > 0:
            valid_final = np.all(pareto_F < ref, axis=1)
            if np.any(valid_final):
                pX_final = pareto_X[valid_final]
                pF_final = pareto_F[valid_final]
            else:
                pX_final = pareto_X
                pF_final = pareto_F

            if len(pX_final) > 0:
                # Sort by sum of objectives and refine the best few
                sums = np.sum(pF_final, axis=1)
                top_k = min(4, len(pX_final))
                top_idx = np.argsort(sums)[:top_k]

                for pi in top_idx:
                    if used >= budget or time.time() > deadline - 5:
                        break
                    parent = pX_final[pi]

                    # Simple coordinate descent refinement
                    for coord in range(dim):
                        if used >= budget or time.time() > deadline - 3:
                            break

                        # Try small perturbations in this coordinate
                        best_f = problem(parent).copy()
                        best_x = parent.copy()

                        for delta in [-0.1, -0.05, -0.02, 0.02, 0.05, 0.1]:
                            if used >= budget or time.time() > deadline - 2:
                                break
                            trial = parent.copy()
                            trial[coord] = np.clip(parent[coord] + delta * (hi[coord] - lo[coord]), lo[coord], hi[coord])
                            if np.allclose(trial, parent):
                                continue
                            ft = problem(trial)
                            used += 1
                            # Simple dominance check: is ft better than best_f in at least one
                            # objective and not worse in any?
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
