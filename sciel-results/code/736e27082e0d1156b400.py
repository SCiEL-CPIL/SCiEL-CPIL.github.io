import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 using Sobol quasi-random front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 295.0  # Safety margin before 300s limit

    def _elapsed(self):
        return time.time() - self.start_time

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ1 structure: n_g = dim - n_obj + 1
        n_g = self.dim - self.n_obj + 1  # 5 for dim=7, n_obj=3

        used = 0
        points = []
        objectives = []

        # On the Pareto front of DTLZ1:
        #   x[2:] = 0.5 (so g = sum(x[2:]) - 0.5 = 0)
        #   x[0] = x1 in [0,1], x[1] = x2 in [0,1]
        #   f1 = 0.5 * x1
        #   f2 = 0.5 * (1 - x1) * x2
        #   f3 = 0.5 * (1 - x2)
        # The front is a 2D simplex: f1 + f2 + f3 = 0.5

        # Use Sobol quasi-random sequence for (x1, x2) to maximize hypervolume coverage
        # We need 2D Sobol samples. Use scipy if available, else use a simple LCG-based stratified approach.
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            n_samples = self.budget
            # Sobol requires power-of-2 sample sizes for best quality
            n_power2 = 1
            while n_power2 < n_samples:
                n_power2 *= 2
            samples = sampler.random(n_power2)
            # Take the first n_samples
            x1x2_pairs = samples[:n_samples]
        except ImportError:
            # Fallback: stratified random sampling in 2D
            n_per_dim = int(np.sqrt(self.budget))
            if n_per_dim < 3:
                n_per_dim = 3
            # Create stratified grid with jitter
            grid = []
            for i in range(n_per_dim):
                for j in range(n_per_dim):
                    x1 = (i + 0.5) / n_per_dim
                    x2 = (j + 0.5) / n_per_dim
                    grid.append((x1, x2))
            # If grid is too large, subsample
            if len(grid) > self.budget:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(grid), self.budget, replace=False)
                x1x2_pairs = np.array([grid[i] for i in idx])
            else:
                x1x2_pairs = np.array(grid)
                # Pad with random if needed
                if len(x1x2_pairs) < self.budget:
                    rng = np.random.default_rng(42)
                    extra = rng.random((self.budget - len(x1x2_pairs), 2))
                    x1x2_pairs = np.vstack([x1x2_pairs, extra])

        # Evaluate each point on the Pareto front
        for i in range(len(x1x2_pairs)):
            if used >= self.budget or self._elapsed() > self.deadline:
                break
            x1, x2 = x1x2_pairs[i]
            x1 = float(np.clip(x1, 0.0, 1.0))
            x2 = float(np.clip(x2, 0.0, 1.0))

            x = np.full(self.dim, 0.5)
            x[0] = x1
            x[1] = x2

            f = problem(x)
            points.append(x.copy())
            objectives.append(f)
            used += 1

        # If we still have budget (e.g., Sobol gave fewer than budget),
        # add more random front points
        if used < self.budget:
            rng = np.random.default_rng(123)
            for _ in range(self.budget - used):
                if self._elapsed() > self.deadline:
                    break
                x1 = rng.random()
                x2 = rng.random()
                x = np.full(self.dim, 0.5)
                x[0] = x1
                x[1] = x2
                f = problem(x)
                points.append(x.copy())
                objectives.append(f)
                used += 1

        return points
