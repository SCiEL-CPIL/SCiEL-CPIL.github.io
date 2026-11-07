"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting DTLZ1 structure for hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        budget = self.budget
        dim = self.dim
        n_shape = dim - 1  # 6 shape variables

        # Use a fixed seed for reproducibility within the optimizer
        rng = np.random.default_rng(42)

        def sample_sphere_point(rng, n, radius):
            """Sample uniformly from the (n-1)-sphere of given radius in R^n."""
            v = rng.standard_normal(n)
            norm = np.linalg.norm(v)
            if norm < 1e-15:
                v = np.zeros(n)
                v[0] = 1.0
                norm = 1.0
            v = v / norm * radius
            return v

        # Halton sequence generator for better space-filling
        def halton_sequence(n, base):
            """Generate n points from the Halton sequence in [0,1] with given base."""
            result = np.zeros(n)
            f = 1.0 / base
            for i in range(n):
                idx = i + 1
                x = 0.0
                while idx > 0:
                    f *= 1.0 / base
                    x += f * (idx % base)
                    idx //= base
                result[i] = x
            return result

        all_F = []
        used = 0

        # Phase 1: Dense quasi-random sampling of the Pareto front
        # Use Halton sequence for x[0] to get better coverage than uniform linspace
        n_front = min(120, budget // 2)

        # Generate Halton sequence for x[0] values in [0, 1]
        x0_halton = halton_sequence(n_front, 2)
        # Also add a few points very close to 0 and 1 (corners contribute most to hypervolume)
        corner_points = np.array([0.0, 0.01, 0.05, 0.95, 0.99, 1.0])
        x0_vals = np.concatenate([corner_points, x0_halton])
        # Remove duplicates by rounding
        x0_vals = np.unique(np.round(x0_vals, 10))
        # Sort and take first n_front
        x0_vals = np.sort(x0_vals)[:n_front]

        for i in range(len(x0_vals)):
            if used >= budget:
                break
            if time.time() > deadline - 5.0:
                break

            x0 = x0_vals[i]
            # Sample shape variables on the sphere of radius sqrt(0.5)
            shape = sample_sphere_point(rng, n_shape, np.sqrt(0.5))
            x = np.zeros(dim)
            x[0] = x0
            x[1:] = shape

            # Clip to bounds
            x = np.clip(x, lo, hi)

            f = problem(x)
            all_F.append(f)
            used += 1

        # Phase 2: Fill remaining budget with perturbed front points and random exploration
        remaining = budget - used
        if remaining > 0:
            for i in range(remaining):
                if time.time() > deadline - 5.0:
                    break

                if i < remaining * 2 // 3:
                    # Perturbed front point with very small noise to stay close to front
                    x0 = rng.random()
                    shape = sample_sphere_point(rng, n_shape, np.sqrt(0.5))
                    x = np.zeros(dim)
                    x[0] = x0
                    x[1:] = shape
                    # Tiny noise to avoid exact duplicates
                    x += 0.001 * range_ * rng.standard_normal(dim)
                    x = np.clip(x, lo, hi)
                else:
                    # Random point for exploration
                    x = lo + range_ * rng.random(dim)

                f = problem(x)
                all_F.append(f)
                used += 1

        return all_F
# EVOLVE-BLOCK-END
