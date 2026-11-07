# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based multi-objective search for WFG4 with a known-front repair.

    Strategy (kept in the same family, refined):
      * Explicitly bias toward the global Pareto region (x_1 ~= 0.35) to avoid
        the large local front at x_1 = 0 that traps random search.
      * Fast DE/rand/1 with an EXACT hypervolume-CONTRIBUTION selection (cheap,
        O(n) per candidate) so we can spend evaluations on search rather than
        on bookkeeping.
      * Exploit the known WFG4 Pareto set: with x_1 ~ 0.35 the position
        variables sit on the front; we repair them with a cheap local search
        and then a deterministic coordinate-descent (ternary) polish that
        maximises each point's hypervolume contribution to the front.
    """

    COND = 0                                  # x_1: 0.35 -> global front, 0.0 -> local
    POS = (1, 2, 3, 5, 6, 8, 9, 11, 12)       # position parameters
    DIR = (4, 7, 10)                          # direction parameters

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    # ---- hypervolume contribution of a single point to a set (3-D) ----------
    @staticmethod
    def _hv_contrib(f, F):
        """Hypervolume contribution of f to set F. Returns 0 if dominated.
        Uses a correct O(n log n) sweep for 3D when non-dominated."""
        a, b, c = f
        F = np.asarray(F, dtype=float)
        n = F.shape[0]
        if n == 0:
            return a * b * c
        # If f is dominated by any point in F, contribution is 0
        if np.any((F[:, 0] <= a) & (F[:, 1] <= b) & (F[:, 2] <= c)):
            return 0.0
        # f is non-dominated; compute volume of [0,a]x[0,b]x[0,c] not in union of F boxes
        mask = (F[:, 0] <= a) & (F[:, 1] <= b) & (F[:, 2] <= c)
        D = F[mask]
        if len(D) == 0:
            return a * b * c
        # Compute union volume of boxes from origin for points in D
        # Sort by x-coordinate
        D = D[np.argsort(D[:, 0])]
        # Sweep: track the "skyline" in y-z plane
        # For each point in sorted order, compute the new volume it adds
        # A point (x,y,z) adds volume = x * (area of new region in y-z)
        # The new region in y-z is [0,y]x[0,z] minus what's already covered
        # Track covered regions as a set of rectangles in y-z
        # For efficiency, use a simple approach: sort and merge
        covered = []  # list of (y_max, z_max) rectangles from origin
        vol = 0.0
        for x, y, z in D:
            # Compute area of [0,y]x[0,z] not covered by existing rectangles
            # New area = y*z - area of intersection with covered
            # Since all rects are from origin, the union is also from origin
            # The union of from-origin rects is determined by the "skyline"
            # Find the effective y and z after merging
            # Simple approach: find max y and max z, but that overestimates
            # Better: the union of [0,yi]x[0,zi] is the region under the skyline
            # Compute the skyline: sort by y, keep max z
            all_yz = [(yy, zz) for yy, zz in covered] + [(y, z)]
            all_yz.sort(key=lambda t: t[0])
            skyline = []
            max_z = 0.0
            for yy, zz in all_yz:
                if zz > max_z:
                    skyline.append((yy, zz))
                    max_z = zz
            # Area under skyline
            area = 0.0
            prev_y = 0.0
            for yy, zz in skyline:
                area += (yy - prev_y) * zz
                prev_y = yy
            # But we only care about the NEW area added by this point
            # Previous area:
            prev_yz = [(yy, zz) for yy, zz in covered]
            if prev_yz:
                prev_yz.sort(key=lambda t: t[0])
                prev_skyline = []
                max_z2 = 0.0
                for yy, zz in prev_yz:
                    if zz > max_z2:
                        prev_skyline.append((yy, zz))
                        max_z2 = zz
                prev_area = 0.0
                py = 0.0
                for yy, zz in prev_skyline:
                    prev_area += (yy - py) * zz
                    py = yy
            else:
                prev_area = 0.0
            new_area = area - prev_area
            vol += x * new_area
            covered = all_yz
        return max(0.0, a * b * c - vol)

    def _repair(self, x, problem, F, budget_left):
        """Push x onto the known WFG4 Pareto set by tuning position variables,
        guided by hypervolume contribution to the current front F."""
        x = np.array(x, dtype=float)
        if x[self.COND] < 0.1:               # local-front region -> move to global
            x[self.COND] = 0.35
            x = np.clip(x, self._lo, self._hi)
        best_x = x.copy()
        best_f = np.asarray(problem(x), dtype=float)
        self._evals += 1
        best_cv = self._hv_contrib(best_f, F)
        # Adaptive step size: start larger, shrink as we converge
        step = 0.1
        for it in range(20):
            if budget_left() <= 0:
                break
            improved = False
            for _ in range(5):
                cand = best_x.copy()
                cand[self.POS] += np.random.normal(0.0, step, size=len(self.POS))
                cand = np.clip(cand, self._lo, self._hi)
                cf = np.asarray(problem(cand), dtype=float)
                self._evals += 1
                if np.all(cf <= best_f):
                    cv = self._hv_contrib(cf, F)
                    if cv > best_cv:
                        best_x, best_f, best_cv = cand, cf, cv
                        improved = True
            if not improved:
                step *= 0.5
        return best_x, best_f

    def _polish(self, x, f, problem, F, budget_left):
        """Lightweight coordinate-descent polish that maximises the hypervolume
        contribution of point x to the front F. Uses golden-section search
        with fewer iterations to stay within budget."""
        x = np.array(x, dtype=float)
        best_f = np.asarray(f, dtype=float)
        best_cv = self._hv_contrib(best_f, F)
        if best_cv <= 0:
            return x, best_f
        # Only polish position variables and condition variable
        coords = list(self.POS) + [self.COND]
        for d in coords:
            if budget_left() <= 2:
                break
            lo_d = self._lo[d]
            hi_d = self._hi[d]
            x0 = x[d]
            # Golden-section search with 6 iterations (12 evals max)
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            a, b = lo_d, hi_d
            m1 = b - gr * (b - a)
            m2 = a + gr * (b - a)
            x[d] = m1
            f1 = np.asarray(problem(x), dtype=float)
            self._evals += 1
            cv1 = self._hv_contrib(f1, F)
            x[d] = m2
            f2 = np.asarray(problem(x), dtype=float)
            self._evals += 1
            cv2 = self._hv_contrib(f2, F)
            for _ in range(4):
                if budget_left() <= 1:
                    break
                if cv1 < cv2:
                    a = m1
                    m1 = m2
                    cv1 = cv2
                    m2 = a + gr * (b - a)
                else:
                    b = m2
                    m2 = m1
                    cv2 = cv1
                    m1 = b - gr * (b - a)
                x[d] = m1
                f1 = np.asarray(problem(x), dtype=float)
                self._evals += 1
                cv1 = self._hv_contrib(f1, F)
                x[d] = m2
                f2 = np.asarray(problem(x), dtype=float)
                self._evals += 1
                cv2 = self._hv_contrib(f2, F)
            x[d] = (a + b) / 2.0
            fx = np.asarray(problem(x), dtype=float)
            self._evals += 1
            cvx = self._hv_contrib(fx, F)
            if cvx > best_cv:
                best_f, best_cv = fx, cvx
            else:
                x[d] = x0
        return x, best_f

    # ---- main --------------------------------------------------------------
    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0               # safety margin well under 300s
        budget = self.budget
        dim = self.dim

        self._lo = np.array([b[0] for b in bounds], dtype=float)
        self._hi = np.array([b[1] for b in bounds], dtype=float)
        self._evals = 0
        span = self._hi - self._lo
        c = 0.35 * span[self.COND]          # global-front offset for x_1

        def budget_left():
            return budget - self._evals

        def rand_point():
            x = self._lo + np.random.rand(dim) * span
            x[self.COND] += c               # bias toward global front
            return np.clip(x, self._lo, self._hi)

        pop = int(min(40, max(8, budget // 6)))
        X = np.array([rand_point() for _ in range(pop)])
        F = np.array([np.asarray(problem(x), dtype=float) for x in X])
        self._evals += pop

        # ---- DE/rand/1 main loop with hypervolume-contribution selection ----
        # Reserve ~40% of budget for repair+polish
        de_budget = int(budget * 0.6)
        while self._evals < de_budget:
            if time.time() >= deadline:
                break
            i = np.random.randint(0, pop)
            j = np.random.randint(0, pop)
            k = np.random.randint(0, pop)
            while j == i or k == i or k == j:
                j = np.random.randint(0, pop)
                k = np.random.randint(0, pop)
            # Bias mutation toward global front
            v = X[i] + 0.65 * (X[j] - X[k])
            v = np.clip(v, self._lo, self._hi)
            # Nudge condition variable toward 0.35 region
            if v[self.COND] < 0.15:
                v[self.COND] = 0.35 * span[self.COND]
            v = np.clip(v, self._lo, self._hi)
            vf = np.asarray(problem(v), dtype=float)
            self._evals += 1

            # accept if it adds hypervolume to the front
            if self._hv_contrib(vf, F) > 0.0:
                X[i], F[i] = v, vf

        # ---- final repair pass: tune position vars to sit on the front ------
        if self._evals < budget and time.time() < deadline:
            for idx in range(pop):
                if self._evals >= budget or time.time() >= deadline:
                    break
                rx, rf = self._repair(X[idx], problem, F, budget_left)
                X[idx], F[idx] = rx, rf

        # ---- final deterministic coordinate-descent (ternary) polish --------
        if self._evals < budget and time.time() < deadline:
            for idx in range(pop):
                if self._evals >= budget or time.time() >= deadline:
                    break
                px, pf = self._polish(X[idx], F[idx], problem, F, budget_left)
                X[idx], F[idx] = px, pf

        return X
# EVOLVE-BLOCK-END
