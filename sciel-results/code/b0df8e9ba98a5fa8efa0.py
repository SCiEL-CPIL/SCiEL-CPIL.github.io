# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34: HV-guided local search with LHS init."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, pts, ref):
        """2D hypervolume. pts: (m,2) array, all < ref. ref: (2,) array."""
        m = len(pts)
        if m == 0:
            return 0.0
        if m == 1:
            return float((ref[0] - pts[0, 0]) * (ref[1] - pts[0, 1]))
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        # Vectorized: for each i, x_next[i] = pts[i+1,0] or ref[0]
        x_next = np.empty(m)
        x_next[:-1] = pts[1:, 0]
        x_next[-1] = ref[0]
        x_cur = pts[:, 0]
        # y_min[i] = min(pts[0..i, 1])
        y_min = np.minimum.accumulate(pts[:, 1])
        widths = x_next - x_cur
        heights = ref[1] - y_min
        hv = np.sum(widths * heights)
        return float(hv)

    def _hv_3d(self, pts, ref):
        """3D hypervolume. pts: (n,3) array, all < ref. ref: (3,) array."""
        n = len(pts)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(ref - pts[0]))
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        x_next = np.empty(n)
        x_next[:-1] = pts[1:, 0]
        x_next[-1] = ref[0]
        x_cur = pts[:, 0]
        widths = x_next - x_cur
        # For each i, compute HV of pts[0..i, 1:3] in 2D
        # This is O(n^2) but with vectorized inner 2D HV
        hvs = np.zeros(n)
        for i in range(n):
            yz = pts[:i + 1, 1:3]
            hvs[i] = self._hv_2d(yz, ref[1:3])
        hv = np.sum(widths * hvs)
        return float(hv)

    def _is_pareto(self, F):
        """Return boolean mask of Pareto-optimal points in F (n x 3)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        # Vectorized dominance check
        # F[i] dominates F[j] if F[i] <= F[j] and F[i] != F[j]
        # Create matrix: D[i,j] = True if i dominates j
        le = np.all(F[:, np.newaxis, :] <= F[np.newaxis, :, :], axis=2)  # (n,n)
        neq = np.any(F[:, np.newaxis, :] < F[np.newaxis, :, :], axis=2)  # (n,n)
        dominates = le & neq
        # Point i is dominated if any j (j != i) dominates i
        dominated = np.any(dominates, axis=0)
        pareto = ~dominated
        return pareto

    def _hv_contributions(self, pareto_F, ref):
        """Compute HV contribution of each point in pareto_F."""
        n = len(pareto_F)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([float(np.prod(ref - pareto_F[0]))])
        
        # Compute total HV
        hv_total = self._hv_3d(pareto_F, ref)
        
        # For each point, compute HV without it
        contribs = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            remaining = pareto_F[mask]
            if len(remaining) == 0:
                hv_without = 0.0
            else:
                hv_without = self._hv_3d(remaining, ref)
            contribs[i] = hv_total - hv_without
        return contribs

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # 330s limit, 10s safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1705.0, 12.0, 0.3])

        # Use global np.random (seeded by evaluator) for reproducibility
        rng = np.random  # Use the global seeded RNG

        # --- Phase 1: Initial sampling ---
        # More LHS for better coverage
        n_lhs = 50
        n_rand = 30
        n_init = min(n_lhs + n_rand, budget)

        X_list = []
        # LHS with better stratification
        lhs_X = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = np.random.permutation(n_lhs)
            # Use midpoints of strata for better coverage
            lhs_X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + 0.5 + np.random.uniform(size=n_lhs) * 0.8) / n_lhs
        lhs_X = np.clip(lhs_X, lo, hi)
        X_list.append(lhs_X)

        # Random
        rand_X = lo + np.random.uniform(size=(n_rand, dim)) * (hi - lo)
        rand_X = np.clip(rand_X, lo, hi)
        X_list.append(rand_X)

        X_init = np.vstack(X_list)
        F_init = np.array([problem(x) for x in X_init])
        used = len(X_init)

        # Store all evaluated points
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
            # Prune to top 30 by HV contribution for faster computation
            if len(pareto_F) > 30:
                contribs = self._hv_contributions(pareto_F, ref)
                top_idx = np.argsort(contribs)[-30:]
                pareto_X = pareto_X[top_idx]
                pareto_F = pareto_F[top_idx]
        else:
            pareto_X = np.empty((0, dim))
            pareto_F = np.empty((0, n_obj))

        # --- Phase 2: Iterative improvement ---
        # Cache HV contributions to avoid recomputation
        hv_contribs_cache = None
        hv_contribs_cache_F = None

        while used < budget:
            if time.time() > deadline:
                break

            progress = used / budget

            # Parent selection
            if len(pareto_F) > 0:
                # Use cached HV contributions if available and still valid
                if hv_contribs_cache is not None and np.array_equal(hv_contribs_cache_F, pareto_F):
                    contribs = hv_contribs_cache
                else:
                    contribs = self._hv_contributions(pareto_F, ref)
                    hv_contribs_cache = contribs
                    hv_contribs_cache_F = pareto_F.copy()

                # Ensure positive
                contribs = np.clip(contribs, 1e-10, None)

                # Weighted selection: favor high-contribution parents
                temp = 1.5 * (1.0 - progress) + 0.3
                weighted = np.power(contribs, 1.0 / temp)
                probs_w = weighted / weighted.sum()
                parent_idx = np.random.choice(len(pareto_F), p=probs_w)
                parent = pareto_X[parent_idx]
            else:
                if len(all_X) > 0:
                    parent = all_X[np.random.randint(len(all_X))]
                else:
                    parent = lo + np.random.uniform(size=dim) * (hi - lo)

            # Adaptive step: larger early, smaller late
            base_step = 0.6 * (1.0 - progress * 0.8) + 0.05
            step = base_step * (0.6 + np.random.uniform() * 0.8)

            # Mutation: 60% Gaussian, 40% coordinate-specific
            if np.random.uniform() < 0.6:
                child = parent + step * (hi - lo) * np.random.standard_normal(dim)
            else:
                child = parent.copy()
                n_perturb = np.random.randint(1, 3)
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
                # Vectorized dominance check against archive
                if len(pareto_F) > 0:
                    le = np.all(pareto_F <= fc, axis=1)
                    neq = np.any(pareto_F < fc, axis=1)
                    dominated = np.any(le & neq)
                else:
                    dominated = False

                if not dominated:
                    pareto_X = np.vstack([pareto_X, [child]])
                    pareto_F = np.vstack([pareto_F, [fc]])
                    # Remove points dominated by the new point (vectorized)
                    le_new = np.all(pareto_F[:-1] >= fc, axis=1)
                    neq_new = np.any(pareto_F[:-1] > fc, axis=1)
                    to_remove = le_new & neq_new
                    if np.any(to_remove):
                        keep = np.ones(len(pareto_F), dtype=bool)
                        keep[:-1][to_remove] = False
                        pareto_X = pareto_X[keep]
                        pareto_F = pareto_F[keep]
                    
                    # Invalidate cache
                    hv_contribs_cache = None
                    hv_contribs_cache_F = None

                # Prune archive if too large
                if len(pareto_F) > 40:
                    p_mask = self._is_pareto(pareto_F)
                    pareto_X = pareto_X[p_mask]
                    pareto_F = pareto_F[p_mask]
                    if len(pareto_F) > 40:
                        contribs = self._hv_contributions(pareto_F, ref)
                        top_idx = np.argsort(contribs)[-40:]
                        pareto_X = pareto_X[top_idx]
                        pareto_F = pareto_F[top_idx]
                        hv_contribs_cache = None
                        hv_contribs_cache_F = None

        # --- Phase 3: Final polish with tight steps on best points ---
        if time.time() < deadline - 5 and used < budget and len(pareto_F) > 0:
            # Get top 5 by HV contribution
            contribs = self._hv_contributions(pareto_F, ref)
            top_idx = np.argsort(contribs)[-5:]
            top_X = pareto_X[top_idx]
            
            polish_budget = min(budget - used, 30)
            for i in range(polish_budget):
                if time.time() > deadline - 2:
                    break
                parent = top_X[i % len(top_X)]
                # Very tight local search
                step = 0.05 * np.random.uniform(0.5, 1.5)
                child = parent + step * (hi - lo) * np.random.standard_normal(dim)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                all_X = np.vstack([all_X, [child]])
                all_F = np.vstack([all_F, [fc]])
                
                if np.all(fc < ref):
                    if len(pareto_F) > 0:
                        le = np.all(pareto_F <= fc, axis=1)
                        neq = np.any(pareto_F < fc, axis=1)
                        dominated = np.any(le & neq)
                    else:
                        dominated = False
                    if not dominated:
                        pareto_X = np.vstack([pareto_X, [child]])
                        pareto_F = np.vstack([pareto_F, [fc]])
                        # Remove dominated
                        if len(pareto_F) > 1:
                            le_new = np.all(pareto_F[:-1] >= fc, axis=1)
                            neq_new = np.any(pareto_F[:-1] > fc, axis=1)
                            to_remove = le_new & neq_new
                            if np.any(to_remove):
                                keep = np.ones(len(pareto_F), dtype=bool)
                                keep[:-1][to_remove] = False
                                pareto_X = pareto_X[keep]
                                pareto_F = pareto_F[keep]

        return all_X

# EVOLVE-BLOCK-END
