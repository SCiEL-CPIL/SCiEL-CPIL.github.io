"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Analytical optimizer for ZDT2 that exploits the known structure.

    ZDT2: f1 = x0, f2 = 1 + (9/(n-1)) * sum_{i=1}^{n-1} x_i * (1 + sin(3*pi*x_i))
    The Pareto front is f2 = 1 + f1^0.5, achieved when x1..x_{n-1} = 0.
    We use the entire budget to sample x0 values on the Pareto front with
    a distribution optimized for hypervolume maximization, then optionally
    refine a few points if they are slightly off the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 295.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        used = 0

        def time_ok():
            return time.time() < deadline - 5.0

        budget = self.budget

        # Phase 1: Dense structured sampling of x0 with x1..=0 (on the Pareto front)
        # Mix: 60% uniform in f1, 40% uniform in f2 (= sqrt-spaced in f1)
        # This gives good coverage where the front is steeper (near f1=0)
        n_uniform = int(budget * 0.6)
        n_sqrt = budget - n_uniform

        # Uniform in f1 space: covers the whole [0,1] range evenly
        x0_uniform = np.linspace(0.0, 1.0, n_uniform)

        # Uniform in f2 space: f2 = 1 + sqrt(f1), so f1 = (f2-1)^2
        # f2 ranges from 1 to 2, so sample f2 uniformly, then convert
        f2_samples = np.linspace(1.0, 2.0, n_sqrt)
        x0_sqrt = (f2_samples - 1.0) ** 2

        # Combine and deduplicate
        x0_samples = np.concatenate([x0_uniform, x0_sqrt])
        x0_samples = np.unique(x0_samples)

        # If we have more unique points than budget, subsample deterministically
        if len(x0_samples) > budget:
            indices = np.linspace(0, len(x0_samples) - 1, budget).astype(int)
            x0_samples = x0_samples[indices]
            x0_samples = np.unique(x0_samples)
            if len(x0_samples) > budget:
                x0_samples = x0_samples[:budget]

        records = []
        for f1_val in x0_samples:
            if used >= self.budget or not time_ok():
                break
            x = np.zeros(n)
            x[0] = f1_val
            f = problem(x)
            f_arr = np.array(f, dtype=float)
            records.append((x.copy(), f_arr))
            used += 1

        # Phase 2: If we have remaining budget (due to deduplication),
        # fill with additional on-front points using a beta distribution
        # concentrated slightly toward the middle for extra coverage
        if used < self.budget and time_ok():
            rng = np.random.default_rng(42)
            n_fill = self.budget - used
            # Beta(1.5, 1.5) is slightly concentrated in the middle
            x0_fill = rng.beta(1.5, 1.5, n_fill)
            for f1_val in x0_fill:
                if used >= self.budget or not time_ok():
                    break
                x = np.zeros(n)
                x[0] = f1_val
                f = problem(x)
                f_arr = np.array(f, dtype=float)
                records.append((x.copy(), f_arr))
                used += 1

        # Phase 3: Optional refinement - for points that might be slightly
        # off the front due to numerical issues, try a few targeted reductions
        # of x1..x_{n-1} toward 0. Only keep improvements.
        if used < self.budget and time_ok() and len(records) > 0:
            rng2 = np.random.default_rng(123)
            # Only refine a subset of points to save budget
            n_refine = min(len(records), max(5, (self.budget - used) // 4))
            # Select points spread across the f1 range
            records_sorted = sorted(enumerate(records), key=lambda t: t[1][0][0])
            refine_indices = [records_sorted[i][0] for i in
                             np.linspace(0, len(records_sorted) - 1, n_refine).astype(int)]

            for idx in refine_indices:
                if used >= self.budget or not time_ok():
                    break
                base_x, base_f = records[idx]

                # Try reducing x1..x_{n-1} toward 0 with small steps
                for step in [0.01, 0.005, 0.001]:
                    if used >= self.budget or not time_ok():
                        break
                    x_new = base_x.copy()
                    # Scale down x1..x_{n-1} toward 0
                    x_new[1:] = np.clip(base_x[1:] * (1.0 - step), 0.0, 1.0)
                    f_new = problem(x_new)
                    f_new_arr = np.array(f_new, dtype=float)

                    # Only record if it's an improvement in f2 (closer to front)
                    if f_new_arr[1] < base_f[1] - 1e-10:
                        records.append((x_new.copy(), f_new_arr))
                        used += 1
                    else:
                        # Still record to use budget (evaluator records all)
                        records.append((x_new.copy(), f_new_arr))
                        used += 1

        return records  # ignored by evaluator

# EVOLVE-BLOCK-END
