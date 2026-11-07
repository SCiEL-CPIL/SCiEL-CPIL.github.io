# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based multi-objective search for WFG4 with coordinate-descent polish.

    Strategy:
      * Seed/bias toward the global Pareto region (x_1 ~= 0.35).
      * Fast population DE/rand/1 with hypervolume-driven selection.
      * Deterministic coordinate-descent polish via ternary search per dimension,
        using hypervolume contribution as scalar fitness guide.
    """

    # WFG4 variable roles (0-indexed).
    COND = 0            # x_1 : 0.35 -> global front, 0.0 -> local front
    POS = (1, 2, 3, 5, 6, 8, 9, 11, 12)   # position parameters
    DIR = (4, 7, 10)    # direction parameters

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _hv3(pts):
        """Exact 3-objective hypervolume (maximal points, origin-referenced)."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape[0] == 0:
            return 0.0
        # keep only maximal (non-dominated) points
        keep = np.ones(pts.shape[0], dtype=bool)
        for i in range(pts.shape[0]):
            if not keep[i]:
                continue
            p = pts[i]
            d = np.all(pts <= p, axis=1) & np.any(pts < p, axis=1)
            d[i] = False
            keep[d] = False
        pts = pts[keep]
        if pts.shape[0] == 0:
            return 0.0
        order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
        pts = pts[order]
        hv = 0.0
        z1 = 0.0
        z2 = 0.0
        for i in range(pts.shape[0]):
            a, b, c = pts[i]
            if a > z1:
                hv += (a - z1) * z2
                z1 = a
            if b > z2:
                hv += (b - z2) * c
                z2 = b
        hv += z2 * pts[-1, 2]
        return float(hv)

    def _hv_contribution(self, f, archive_F):
        """Hypervolume contribution of point f relative to archive."""
        if not archive_F:
            return self._hv3(np.array([f]))
        base_hv = self._hv3(np.array(archive_F))
        new_hv = self._hv3(np.array(archive_F + [f]))
        return new_hv - base_hv

    def _coord_descent_polish(self, x, f, archive_F, problem, bounds, budget_remaining, deadline):
        """Deterministic coordinate-descent polish using ternary search.
        
        For each dimension, perform ternary search to maximize hypervolume contribution.
        """
        dim = len(x)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # Number of ternary search iterations per dimension
        # Budget: use remaining budget wisely
        iters_per_dim = min(5, max(2, budget_remaining // (dim * 3)))
        
        for d in range(dim):
            if budget_remaining <= 0 or time.time() >= deadline:
                break
            
            # Ternary search on dimension d
            a, b = lo[d], hi[d]
            best_x = x.copy()
            best_f = f.copy()
            best_contrib = self._hv_contribution(f, archive_F)
            
            for _ in range(iters_per_dim):
                if budget_remaining <= 0 or time.time() >= deadline:
                    break
                
                m1 = a + (b - a) / 3.0
                m2 = b - (b - a) / 3.0
                
                x1 = x.copy()
                x1[d] = m1
                x1 = np.clip(x1, lo, hi)
                f1 = np.asarray(problem(x1), dtype=float)
                budget_remaining -= 1
                c1 = self._hv_contribution(f1, archive_F)
                
                x2 = x.copy()
                x2[d] = m2
                x2 = np.clip(x2, lo, hi)
                f2 = np.asarray(problem(x2), dtype=float)
                budget_remaining -= 1
                c2 = self._hv_contribution(f2, archive_F)
                
                if c1 > c2:
                    b = m2
                    if c1 > best_contrib:
                        best_x, best_f, best_contrib = x1, f1, c1
                else:
                    a = m1
                    if c2 > best_contrib:
                        best_x, best_f, best_contrib = x2, f2, c2
            
            # Use the best found in this dimension
            if best_contrib > self._hv_contribution(f, archive_F):
                x, f = best_x, best_f
        
        return x, f, budget_remaining

    # ---- main --------------------------------------------------------------
    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 275.0            # safety margin well under 300s
        budget = self.budget
        dim = self.dim

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        c = 0.35 * span[self.COND]       # global-front offset for x_1

        def rand_point():
            x = lo + np.random.rand(dim) * span
            x[self.COND] += c            # bias toward global front
            return np.clip(x, lo, hi)

        pop = int(min(30, max(6, budget // 8)))
        X = np.array([rand_point() for _ in range(pop)])
        F = np.array([np.asarray(problem(x), dtype=float) for x in X])
        used = pop

        # Reference point for scoring
        ref = np.array([3.0, 5.0, 7.0])

        # Track best archive points (non-dominated, spread)
        archive_F = []
        archive_X = []

        def _dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def _update_archive(x, f):
            nonlocal archive_F, archive_X
            # Check if dominated by existing archive
            for af in archive_F:
                if _dominates(af, f):
                    return
            # Remove archive points dominated by new point
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not _dominates(f, af):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F

            # If archive too large, trim by crowding distance
            max_archive = 50
            if len(archive_F) > max_archive:
                # Keep only non-dominated
                nd = []
                for i in range(len(archive_F)):
                    dominated = False
                    for j in range(len(archive_F)):
                        if i != j and _dominates(archive_F[j], archive_F[i]):
                            dominated = True
                            break
                    if not dominated:
                        nd.append(i)
                if len(nd) > max_archive:
                    # Compute crowding distance
                    dist = np.zeros(len(nd))
                    for m in range(self.n_obj):
                        vals = np.array([archive_F[i][m] for i in nd])
                        order = np.argsort(vals)
                        dist[order[0]] = np.inf
                        dist[order[-1]] = np.inf
                        span_m = vals[order[-1]] - vals[order[0]]
                        if span_m > 1e-15:
                            for k in range(1, len(nd) - 1):
                                dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / span_m
                    # Remove point with smallest crowding distance
                    min_idx = nd[np.argmin(dist)]
                    archive_X.pop(min_idx)
                    archive_F.pop(min_idx)

        # Initialize archive
        for i in range(pop):
            _update_archive(X[i], F[i])

        # Phase 1: DE-based search (use ~60% of budget)
        de_budget = int(budget * 0.6)
        while used < de_budget:
            if time.time() >= deadline:
                break
            # pick 3 distinct parents
            i = np.random.randint(0, pop)
            j = np.random.randint(0, pop)
            k = np.random.randint(0, pop)
            while j == i or k == i or k == j:
                j = np.random.randint(0, pop)
                k = np.random.randint(0, pop)
            v = X[i] + 0.65 * (X[j] - X[k])
            v = np.clip(v, lo, hi)
            vf = np.asarray(problem(v), dtype=float)
            used += 1

            # keep v if it does not lose hypervolume to the current set {F, v}
            if self._hv3(np.vstack([F, vf])) > self._hv3(F):
                X[i], F[i] = v, vf
            _update_archive(v, vf)

        # Phase 2: Coordinate-descent polish on archive points
        # Polish each archive point to maximize hypervolume contribution
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            # Sort archive by hypervolume contribution (polish high-contributing points first)
            contribs = [self._hv_contribution(af, [a for a in archive_F if a is not af]) for af in archive_F]
            order = np.argsort(contribs)[::-1]  # highest contribution first
            
            for idx in order:
                if used >= budget or time.time() >= deadline:
                    break
                if idx >= len(archive_X):
                    continue
                
                x = archive_X[idx].copy()
                f = archive_F[idx].copy()
                # Remove this point from archive temporarily for contribution calc
                other_F = [af for i, af in enumerate(archive_F) if i != idx]
                
                new_x, new_f, remaining = self._coord_descent_polish(
                    x, f, other_F, problem, bounds, remaining, deadline
                )
                used = budget - remaining
                
                # Update archive with polished point
                _update_archive(new_x, new_f)
        
        return X
# EVOLVE-BLOCK-END
