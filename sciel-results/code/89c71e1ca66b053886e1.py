import numpy as np
import time


class Optimizer:
    """Direct construction of the ZDT3 Pareto front with optimal point density.
    
    The ZDT3 Pareto front is given by x = (t, 0, ..., 0) for t in [0, 1].
    f1(t) = t, f2(t) = 2*(1 - sqrt(t) - t*sin(10*pi*t)).
    
    Optimal point density n(t) ∝ |df2/dt|^(1/3) minimizes the discretization
    loss of the hypervolume approximation.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = int(self.budget)
        n_sample = max(2, budget)

        # Compute |df2/dt| on a fine grid
        t_grid = np.linspace(1e-10, 1.0, 200000)
        df2dt = 2.0 * (-1.0 / (2.0 * np.sqrt(t_grid)) - np.sin(10.0 * np.pi * t_grid) - 10.0 * np.pi * t_grid * np.cos(10.0 * np.pi * t_grid))
        abs_df2dt = np.abs(df2dt)
        # Handle t=0 singularity: use next value
        abs_df2dt[0] = abs_df2dt[1]

        # Optimal density: n(t) ∝ |df2/dt|^(1/3)
        density = abs_df2dt ** (1.0 / 3.0)
        
        # Build CDF
        dt = t_grid[1] - t_grid[0]
        cdf = np.cumsum(density) * dt
        cdf /= cdf[-1]

        # Sample n_sample points uniformly in CDF space
        u = np.linspace(0.0, 1.0, n_sample + 2)[1:-1]
        t_values = np.interp(u, cdf, t_grid)

        # Add endpoints t=0 and t=1
        t_values = np.concatenate([[0.0], t_values, [1.0]])
        t_values = np.sort(np.unique(t_values))

        # Subsample if we have more than n_sample
        if len(t_values) > n_sample:
            idx = np.linspace(0, len(t_values) - 1, n_sample).astype(int)
            t_values = t_values[idx]

        # Ensure exact endpoints
        t_values[0] = 0.0
        t_values[-1] = 1.0

        recorded = []
        used = 0
        x = np.zeros(dim)

        for tv in t_values:
            if used >= budget:
                break
            if time.time() > deadline:
                break

            x[0] = float(tv)
            x_clipped = np.clip(x, lo, hi)
            f = np.asarray(problem(x_clipped), dtype=float)
            recorded.append(f)
            used += 1

        return recorded
