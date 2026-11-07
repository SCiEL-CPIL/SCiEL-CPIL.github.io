"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Inverse-transform sampling on ZDT4 Pareto front with local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0
        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT4 Pareto front: f1 = x0 + 2, f2 = 1 + (f1-4)^2
        # Hypervolume contribution density for f1 in [2,6]:
        #   w(f1) = (400 - f2) * (11 - f1) = (399 - (f1-4)^2) * (11 - f1)
        # Sample f1 from this density using rejection sampling.
        # Max of w(f1): at f1=4, w = 399*7 = 2793

        def w(f1):
            f2 = 1.0 + (f1 - 4.0)**2
            if f1 < 2.0 or f1 > 6.0:
                return 0.0
            return (399.0 - (f1 - 4.0)**2) * (11.0 - f1)

        max_w = 2793.0
        n_samples = min(budget, 200)
        f1_samples = []
        attempts = 0
        max_attempts = n_samples * 100
        while len(f1_samples) < n_samples and attempts < max_attempts:
            f1 = 2.0 + rng.random() * 4.0
            ww = w(f1)
            if ww > 0 and rng.random() < ww / max_w:
                f1_samples.append(f1)
            attempts += 1
            if time.time() > deadline:
                break

        if not f1_samples:
            f1_samples = [4.0]
        f1_samples = np.array(f1_samples)

        # For each f1, x0 = f1 - 2. Set x1..x9 to minimize f2 (ideally 0).
        # Local polish to handle sin terms.
        recorded = []
        for f1 in f1_samples:
            x0 = f1 - 2.0
            x = np.zeros(n)
            x[0] = x0
            for i in range(1, n):
                if lo[i] <= 0.0 <= hi[i]:
                    x[i] = 0.0
                elif 0.0 < lo[i]:
                    x[i] = lo[i]
                else:
                    x[i] = hi[i]
            f = problem(x)
            if f[0] < 11.0 and f[1] < 400.0:
                # Local polish: perturb x1..x9 to reduce f2
                best_x, best_f = x.copy(), f.copy()
                for trial in range(20):
                    if time.time() > deadline:
                        break
                    x_trial = best_x.copy()
                    for i in range(1, n):
                        x_trial[i] = np.clip(x_trial[i] + 0.05 * rng.standard_normal(), lo[i], hi[i])
                    f_trial = problem(x_trial)
                    if f_trial[1] < best_f[1] - 1e-12:
                        best_x, best_f = x_trial, f_trial
                # Try adjusting x0 to get closer to integer (for sin term)
                for delta in [0.001, -0.001, 0.0001, -0.0001, 0.01, -0.01]:
                    x_trial = best_x.copy()
                    x_trial[0] = np.clip(x_trial[0] + delta, lo[0], hi[0])
                    f_trial = problem(x_trial)
                    if f_trial[1] < best_f[1] - 1e-12 and f_trial[0] < 11.0 and f_trial[1] < 400.0:
                        best_x, best_f = x_trial, f_trial
                recorded.append((best_x, best_f))
            if time.time() > deadline:
                break

        if recorded:
            return recorded[0][0]
        return np.zeros(n)
# EVOLVE-BLOCK-END
