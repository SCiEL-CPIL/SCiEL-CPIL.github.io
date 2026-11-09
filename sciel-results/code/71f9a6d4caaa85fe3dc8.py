"""Multi-objective optimizer for RE34 using Sobol initialization and HV-guided local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer using Sobol initialization and HV-guided local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _hv_3d(self, F, ref):
        """Compute exact hypervolume for 3D points strictly below ref."""
        valid = np.all(F < ref, axis=1)
        Fv = F[valid]
        n = len(Fv)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(np.maximum(0.0, ref - Fv[0])))

        # Sort by f1 ascending
        order = np.argsort(Fv[:, 0])
        Fv = Fv[order]
        n = len(Fv)

        # Remove dominated points
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if Fv[j, 1] >= Fv[i, 1] and Fv[j, 2] >= Fv[i, 2]:
                    keep[j] = False
        Fv = Fv[keep]
        n = len(Fv)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(np.maximum(0.0, ref - Fv[0])))

        # Sort by f1 ascending again
        Fv = Fv[np.argsort(Fv[:, 0])]
        n = len(Fv)

        hv = 0.0
        skyline = []  # list of [f2, f3], sorted by f2 asc, f3 strictly decreasing

        for i in range(n):
            f1_i = Fv[i, 0]
            f1_next = Fv[i + 1, 0] if i + 1 < n else ref[0]
            f1_next = min(f1_next, ref[0])

            if f1_next <= f1_i:
                continue

            f2_i = Fv[i, 1]
            f3_i = Fv[i, 2]

            # Update skyline
            new_skyline = []
            for pt in skyline:
                if not (pt[0] >= f2_i and pt[1] >= f3_i):
                    new_skyline.append(pt)
            new_skyline.append([f2_i, f3_i])
            new_skyline.sort(key=lambda x: x[0])
            final_skyline = []
            for pt in new_skyline:
                if not final_skyline or pt[1] < final_skyline[-1][1]:
                    final_skyline.append(pt)
            skyline = final_skyline

            # Compute 2D HV of skyline
            hv_2d = 0.0
            for j in range(len(skyline)):
                f2_j = skyline[j][0]
                f3_j = skyline[j][1]
                if f2_j >= ref[1] or f3_j >= ref[2]:
                    continue
                f2_next = skyline[j + 1][0] if j + 1 < len(skyline) else ref[1]
                f2_next = min(f2_next, ref[1])
                if f2_next > f2_j:
                    hv_2d += (f2_next - f2_j) * (ref[2] - f3_j)

            hv += (f1_next - f1_i) * hv_2d

        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        ref = np.array([1705.0, 12.0, 0.3])

        X_list = []
        F_list = []

        # Phase 1: Sobol initialization (128 evaluations)
        n_sobol = min(128, self.budget // 2)
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=0)
        sobol_samples = sampler.random(n=n_sobol)
        # Map from [0,1]^dim to [lo, hi]
        sobol_x = lo + sobol_samples * span
        for i in range(n_sobol):
            f = np.array(problem(sobol_x[i]))
            X_list.append(sobol_x[i])
            F_list.append(f)

        used = n_sobol
        step_size = 0.15

        # Phase 2: HV-guided local search for remaining budget
        while used < self.budget:
            if time.time() > deadline:
                break

            F_arr = np.array(F_list)
            n = len(F_arr)

            # Compute exact HV contribution for each point
            valid = np.all(F_arr < ref, axis=1)
            n_valid = np.sum(valid)

            if n_valid == 0:
                parent_idx = rng.integers(n)
            else:
                # Compute full HV
                full_hv = self._hv_3d(F_arr, ref)

                if full_hv <= 0:
                    parent_idx = rng.integers(n)
                else:
                    # For each valid point, estimate its contribution
                    valid_indices = np.where(valid)[0]

                    # Quick screening: points with better f1 values tend to contribute more
                    Fv = F_arr[valid_indices]
                    order = np.argsort(Fv[:, 0])
                    top_candidates = valid_indices[order[:min(20, n_valid)]]

                    # Compute exact contribution for top candidates
                    contrib = np.zeros(n)
                    for idx in top_candidates:
                        mask = np.ones(n, dtype=bool)
                        mask[idx] = False
                        F_without = F_arr[mask]
                        hv_without = self._hv_3d(F_without, ref)
                        contrib[idx] = full_hv - hv_without

                    # Also include some random valid points for diversity
                    if n_valid > len(top_candidates):
                        extra = rng.choice(valid_indices, size=min(5, n_valid - len(top_candidates)), replace=False)
                        for idx in extra:
                            mask = np.ones(n, dtype=bool)
                            mask[idx] = False
                            F_without = F_arr[mask]
                            hv_without = self._hv_3d(F_without, ref)
                            contrib[idx] = full_hv - hv_without

                    # Select parent weighted by contribution
                    pos_contrib = contrib > 0
                    if np.sum(pos_contrib) > 0:
                        weights = contrib[pos_contrib]
                        weights = weights / weights.sum()
                        candidate_idx = np.where(pos_contrib)[0]
                        parent_idx = rng.choice(candidate_idx, p=weights)
                    else:
                        parent_idx = valid_indices[0]

            parent = X_list[parent_idx]

            # Axis-aligned perturbation with small steps
            child = parent.copy()
            # Choose 1-2 axes to perturb
            n_perturb = rng.integers(1, 3)
            axes = rng.choice(self.dim, size=n_perturb, replace=False)
            for ax in axes:
                delta = step_size * span[ax] * rng.standard_normal()
                child[ax] += delta
            child = np.clip(child, lo, hi)

            fc = np.array(problem(child))
            X_list.append(child)
            F_list.append(fc)
            used += 1

            # Adapt step size
            if np.all(fc < ref):
                old_F = np.array(F_list[:-1])
                dominated = np.any(np.all(old_F <= fc, axis=1) & np.any(old_F < fc, axis=1))
                if not dominated:
                    step_size = max(0.05, step_size * 0.92)
                else:
                    step_size = min(1.0, step_size * 1.08)
            else:
                step_size = min(1.0, step_size * 1.15)

        return X_list
# EVOLVE-BLOCK-END
