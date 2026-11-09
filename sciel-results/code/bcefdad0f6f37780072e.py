"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-Driven Multi-Objective Evolutionary Strategy."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        """Return True if a dominates b (all <=, at least one <)."""
        return np.all(a <= b) and np.any(a < b)

    def _is_nondominated(self, point, archive_f):
        """Check if point is non-dominated w.r.t. list of objective vectors."""
        for a in archive_f:
            if self._dominates(a, point):
                return False
        return True

    def _compute_hv_2d(self, points, ref2, ref3):
        """Compute 2D hypervolume for points in (f2, f3) plane."""
        if not points:
            return 0.0
        valid = [(p[0], p[1]) for p in points if p[0] < ref2 and p[1] < ref3]
        if not valid:
            return 0.0
        valid.sort(key=lambda p: p[0])
        hv = 0.0
        n = len(valid)
        for i in range(n):
            f2_next = valid[i + 1][0] if i + 1 < n else ref2
            f2_width = f2_next - valid[i][0]
            if f2_width <= 0:
                continue
            f3_height = ref3 - valid[i][1]
            if f3_height > 0:
                hv += f2_width * f3_height
        return hv

    def _compute_hv(self, points, ref):
        """Compute exact hypervolume for 3D set of points."""
        if not points:
            return 0.0
        valid = [(p[0], p[1], p[2]) for p in points
                 if p[0] < ref[0] and p[1] < ref[1] and p[2] < ref[2]]
        if not valid:
            return 0.0
        valid.sort(key=lambda p: p[0])
        n_valid = len(valid)
        hv = 0.0
        for i in range(n_valid):
            f1_next = valid[i + 1][0] if i + 1 < n_valid else ref[0]
            f1_width = f1_next - valid[i][0]
            if f1_width <= 0:
                continue
            sub_points = [valid[j][1:3] for j in range(i, n_valid)
                          if valid[j][1] < ref[1] and valid[j][2] < ref[2]]
            hv_2d = self._compute_hv_2d(sub_points, ref[1], ref[2])
            hv += f1_width * hv_2d
        return hv

    def _hv_contribution(self, f, archive_f, ref):
        """Compute hypervolume contribution of point f to archive_f."""
        for a in archive_f:
            if self._dominates(a, f):
                return 0.0
        hv_before = self._compute_hv(archive_f, ref)
        hv_after = self._compute_hv(archive_f + [f], ref)
        return hv_after - hv_before

    def _add_to_archive(self, x, f, archive):
        """Add (x, f) to archive if f is non-dominated."""
        archive_f = [a[1] for a in archive]
        if not self._is_nondominated(f, archive_f):
            return archive
        new_archive = []
        for x_a, f_a in archive:
            if not self._dominates(f, f_a):
                new_archive.append((x_a, f_a))
        new_archive.append((x, f))
        return new_archive

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        ref = np.array([3.0, 5.0, 7.0])
        total_budget = self.budget
        used = 0

        archive = []

        def evaluate(x):
            nonlocal used
            if used >= total_budget:
                return None
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            used += 1
            return f

        # Phase 1: Random Initialization
        n_init = min(30, total_budget // 6)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = lo + range_ * rng.random(self.dim)
            f = evaluate(x)
            if f is None:
                break
            archive = self._add_to_archive(x, f, archive)

        if not archive:
            x = (lo + hi) / 2
            f = evaluate(x)
            if f is not None:
                archive.append((x, f))

        # Phase 2: Hypervolume-Driven Evolutionary Loop
        while used < total_budget and time.time() < deadline:
            if len(archive) > 1:
                hv_contribs = []
                archive_f = [a[1] for a in archive]
                for i, (x_a, f_a) in enumerate(archive):
                    others = [archive_f[j] for j in range(len(archive_f)) if j != i]
                    contrib = self._hv_contribution(f_a, others, ref)
                    hv_contribs.append(contrib)
                total_contrib = sum(hv_contribs)
                if total_contrib > 0:
                    probs = np.array(hv_contribs) / total_contrib
                    idx = rng.choice(len(archive), p=probs)
                else:
                    idx = rng.integers(len(archive))
            else:
                idx = 0

            x_parent = archive[idx][0]
            progress = used / total_budget
            base_scale = 0.15 * (1.0 - progress) + 0.02
            noise = rng.standard_t(df=3, size=self.dim)
            noise = np.clip(noise, -3, 3)
            x_child = x_parent + base_scale * range_ * noise
            x_child = np.clip(x_child, lo, hi)

            f_child = evaluate(x_child)
            if f_child is None:
                break
            archive = self._add_to_archive(x_child, f_child, archive)

            # Local search on best HV-contributing point
            if used < total_budget - 10:
                archive_f = [a[1] for a in archive]
                best_idx = 0
                best_contrib = -1
                for i, (x_a, f_a) in enumerate(archive):
                    others = [archive_f[j] for j in range(len(archive_f)) if j != i]
                    contrib = self._hv_contribution(f_a, others, ref)
                    if contrib > best_contrib:
                        best_contrib = contrib
                        best_idx = i

                x_cur = archive[best_idx][0].copy()
                for ls_iter in range(4):
                    if used >= total_budget or time.time() > deadline:
                        break
                    ls_scale = 0.03 * (1.0 - ls_iter * 0.2)
                    x_trial = x_cur + ls_scale * range_ * rng.standard_normal(self.dim)
                    x_trial = np.clip(x_trial, lo, hi)
                    f_trial = evaluate(x_trial)
                    if f_trial is None:
                        break
                    if self._is_nondominated(f_trial, [a[1] for a in archive]):
                        archive = self._add_to_archive(x_trial, f_trial, archive)
                        x_cur = x_trial

        return [a[1] for a in archive]
# EVOLVE-BLOCK-END
