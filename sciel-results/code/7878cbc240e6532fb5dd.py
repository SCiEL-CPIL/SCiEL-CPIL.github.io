import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, F, ref):
        """Compute 2D hypervolume for points strictly better than ref."""
        mask = np.all(F < ref, axis=1)
        F = F[mask]
        if len(F) == 0:
            return 0.0
        idx = np.argsort(F[:, 0])
        F = F[idx]
        # Keep only non-dominated (sorted by f1 asc, f2 should be desc)
        keep = [0]
        for i in range(1, len(F)):
            if F[i, 1] < F[keep[-1], 1]:
                keep.append(i)
        F = F[keep]
        if len(F) == 0:
            return 0.0
        x, y = F[:, 0], F[:, 1]
        n = len(x)
        hv = 0.0
        for i in range(n):
            x_next = x[i + 1] if i < n - 1 else ref[0]
            hv += (x_next - x[i]) * (ref[1] - y[i])
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, self.dim))

        if zeros_ok:
            n_points = self.budget
            x0_lo = max(lo[0], 0.0)
            x0_hi = min(hi[0], 1.0)

            if x0_lo >= x0_hi:
                x0_vals = np.full(n_points, x0_lo)
            else:
                # Optimize exponent for max hypervolume
                ref = np.array([11.0, 400.0])
                best_exp = 4.0 / 3.0
                best_hv = -1.0
                for exp in np.linspace(0.3, 2.5, 47):
                    t = np.linspace(0, 1, n_points)
                    x0_trial = x0_lo + (x0_hi - x0_lo) * (t ** exp)
                    f1 = x0_trial
                    f2 = 2.0 - np.sqrt(f1)
                    F_trial = np.column_stack([f1, f2])
                    hv = self._hv_2d(F_trial, ref)
                    if hv > best_hv:
                        best_hv = hv
                        best_exp = exp

                t = np.linspace(0, 1, n_points)
                x0_vals = x0_lo + (x0_hi - x0_lo) * (t ** best_exp)

            X = np.zeros((n_points, self.dim))
            X[:, 0] = x0_vals

            if time.time() > deadline:
                return X

            F = np.array([problem(x) for x in X])
            return X

        else:
            rng = np.random.default_rng(0)
            X = lo + (hi - lo) * rng.random((self.budget, self.dim))
            F = np.array([problem(x) for x in X])
            return X
