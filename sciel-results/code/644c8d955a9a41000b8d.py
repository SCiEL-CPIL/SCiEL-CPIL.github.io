"""Hypervolume-driven sequential MO optimizer for WFG4 (3 objectives, 12 vars)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s kill

        dim = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        ref = np.array([3.0, 5.0, 7.0])

        used = 0

        def hv_contrib(f):
            """Exact hypervolume contribution of a single point to the current archive (3D)."""
            if f[0] >= ref[0] or f[1] >= ref[1] or f[2] >= ref[2]:
                return 0.0
            # Points that are strictly better than f in at least one objective
            # (and not worse in any) bound the region where f is not the minimizer.
            better = np.where((archive_F < f).any(axis=1))[0]
            if len(better) == 0:
                return (ref[0] - f[0]) * (ref[1] - f[1]) * (ref[2] - f[2])
            # For each objective, the max objective value among better points (capped at ref)
            # The region where f is the unique minimizer is the box from f to (min over better of each coord).
            # Actually: f contributes the box [f_i, min_{j better} f_j_i] intersected with [f_i, ref_i].
            # The "min over better" per coordinate gives the upper bound of the box.
            upper = np.minimum(ref, archive_F[better].min(axis=0))
            # If for any coordinate upper <= f, contribution is 0
            if (upper <= f).any():
                return 0.0
            return float(np.prod(upper - f))

        def add_to_archive(x, f):
            """Add point to archive if not dominated by existing; remove points it dominates."""
            nonlocal used
            # Check if f is dominated by any existing
            for i in range(len(archive_F)):
                if (archive_F[i] <= f).all() and (archive_F[i] < f).any():
                    return  # dominated, discard
            # Remove points dominated by f
            new_idx = []
            for i in range(len(archive_F)):
                if (f <= archive_F[i]).all() and (f < archive_F[i]).any():
                    continue
                new_idx.append(i)
            # Also remove exact duplicates
            for i in new_idx[:]:
                if np.allclose(archive_F[i], f, atol=1e-12):
                    new_idx.remove(i)
            if len(new_idx) == 0:
                archive_F = [f.copy()]
                archive_X = [x.copy()]
            else:
                archive_F = [archive_F[i].copy() for i in new_idx]
                archive_X = [archive_X[i].copy() for i in new_idx]
                archive_F.append(f.copy())
                archive_X.append(x.copy())
            return True

        # --- Initialization: small random + center-biased population ---
        init_size = min(12, self.budget // 20)
        archive_X = []
        archive_F = []
        for _ in range(init_size):
            r = np.random.rand(dim)
            x = lo + span * r
            f = np.asarray(problem(x), dtype=float)
            used += 1
            add_to_archive(x, f)
            if time.time() > deadline:
                break

        # --- Main loop: hypervolume-driven candidate selection ---
        # Candidate pool size: balance between selection quality and time
        n_cand = 40
        n_perturb = 8  # perturbations per archive point

        while used < self.budget:
            if time.time() > deadline:
                break

            # Build candidate pool
            candidates = []

            # 1) Perturbed archive points (exploit near known good regions)
            if len(archive_X) > 0:
                for i in range(len(archive_X)):
                    for _ in range(n_perturb):
                        noise = np.random.randn(dim)
                        # Adaptive step: larger early
                        progress = used / self.budget
                        step = 0.3 * (1 - progress) + 0.05
                        c = archive_X[i] + step * span * noise
                        c = np.clip(c, lo, hi)
                        candidates.append(c)

            # 2) Random points (explore)
            n_random = max(5, n_cand - len(candidates))
            candidates.extend([lo + span * np.random.rand(dim) for _ in range(n_random)])

            # 3) A few structured probes: corners of the box
            corners = []
            for _ in range(3):
                c = np.where(np.random.rand(dim) > 0.5, hi, lo)
                corners.append(c)
            candidates.extend(corners)

            # Evaluate candidates and pick the one with best marginal hypervolume
            best_contrib = -1.0
            best_x = None
            best_f = None

            for c in candidates:
                if time.time() > deadline:
                    break
                f = np.asarray(problem(c), dtype=float)
                used += 1
                contrib = hv_contrib(f)
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_x = c
                    best_f = f

            if best_x is not None:
                add_to_archive(best_x, best_f)

        return None
# EVOLVE-BLOCK-END
