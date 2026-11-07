"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Coordinate descent with adaptive 1D golden-section search,
    exploiting the anisotropy of the Bent Cigar function.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self._eval_count = 0

    def _eval(self, func, x):
        """Evaluate func(x) and track the count."""
        self._eval_count += 1
        return func(x)

    def _golden_section_1d(self, func, x, i, max_iter):
        """
        Perform golden-section search along dimension i of x,
        keeping other coordinates fixed.
        Returns (best_x, best_f, evals_used).
        """
        if self._eval_count >= self.budget:
            return x, np.inf, 0

        golden_ratio = (np.sqrt(5.0) - 1.0) / 2.0

        # Evaluate at current point
        f_orig = self._eval(func, x)

        # Try small perturbations to find direction
        step = 1.0
        x_plus = x.copy()
        x_plus[i] = np.clip(x_plus[i] + step, -5.0, 5.0)
        f_plus = self._eval(func, x_plus)

        if self._eval_count >= self.budget:
            return x, f_orig, 0

        x_minus = x.copy()
        x_minus[i] = np.clip(x_minus[i] - step, -5.0, 5.0)
        f_minus = self._eval(func, x_minus)

        # Choose best starting point
        candidates = [
            (x.copy(), f_orig),
            (x_plus, f_plus),
            (x_minus, f_minus),
        ]
        candidates.sort(key=lambda t: t[1])
        best_x, best_f = candidates[0]

        # Set up bracket
        vals = [t[0][i] for t in candidates]
        v_min, v_max = min(vals), max(vals)
        bracket_size = max((v_max - v_min) * 1.5, step)

        a = max(-5.0, best_x[i] - bracket_size)
        b = min(5.0, best_x[i] + bracket_size)

        if b - a < 1e-4:
            return best_x, best_f, 0

        # Golden section search
        c = a + (1.0 - golden_ratio) * (b - a)
        d = a + golden_ratio * (b - a)

        x_c = best_x.copy()
        x_c[i] = c
        f_c = self._eval(func, x_c)

        if self._eval_count >= self.budget:
            return best_x, best_f, 0

        x_d = best_x.copy()
        x_d[i] = d
        f_d = self._eval(func, x_d)

        # Track best
        if f_c < best_f:
            best_x, best_f = x_c, f_c
        if f_d < best_f:
            best_x, best_f = x_d, f_d

        for _ in range(max_iter):
            if self._eval_count >= self.budget:
                break
            if b - a < 1e-6:
                break

            if f_c < f_d:
                b = d
                d = c
                f_d = f_c
                c = a + (1.0 - golden_ratio) * (b - a)
                x_c = best_x.copy()
                x_c[i] = c
                f_c = self._eval(func, x_c)
                if f_c < best_f:
                    best_x, best_f = x_c, f_c
            else:
                a = c
                c = d
                f_c = f_d
                d = a + golden_ratio * (b - a)
                x_d = best_x.copy()
                x_d[i] = d
                f_d = self._eval(func, x_d)
                if f_d < best_f:
                    best_x, best_f = x_d, f_d

        return best_x, best_f, 0

    def __call__(self, func):
        self._eval_count = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial point: try a few random starts
        n_starts = min(5, budget // 20)
        if n_starts < 1:
            n_starts = 1

        best_x = None
        best_f = np.inf

        for _ in range(n_starts):
            if self._eval_count >= budget:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = self._eval(func, x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = rng.uniform(-5.0, 5.0, dim)
            best_f = self._eval(func, best_x)

        # Coordinate descent
        max_iter_per_search = 12
        no_improve_passes = 0
        max_no_improve = 3

        while self._eval_count < budget and no_improve_passes < max_no_improve:
            improved = False
            for i in range(dim):
                if self._eval_count >= budget:
                    break

                x_new, f_new, _ = self._golden_section_1d(
                    func, best_x, i, max_iter_per_search
                )

                if f_new < best_f - 1e-12:
                    best_x = x_new
                    best_f = f_new
                    improved = True

            if not improved:
                no_improve_passes += 1
                # Try a random perturbation
                if self._eval_count < budget:
                    x_pert = best_x + rng.normal(0, 0.5, dim)
                    x_pert = np.clip(x_pert, -5.0, 5.0)
                    f_pert = self._eval(func, x_pert)
                    if f_pert < best_f:
                        best_x = x_pert
                        best_f = f_pert
                        improved = True
                        no_improve_passes = 0
            else:
                no_improve_passes = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
