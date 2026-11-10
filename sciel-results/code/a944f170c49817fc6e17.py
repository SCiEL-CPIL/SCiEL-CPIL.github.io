"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG9 exploiting its 1D Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.RandomState(42)

        evaluated = []
        used = 0

        # WFG9: Pareto front depends primarily on x[0].
        # Fix other variables to midpoints (they are "position" variables).
        base_x = np.zeros(self.dim)
        for i in range(self.dim):
            base_x[i] = (lo[i] + hi[i]) / 2.0

        # Phase 1: Dense 1D sampling along x[0]
        # Use ~180 evaluations for dense grid on x[0]
        n_grid = 180
        x0_vals = np.linspace(lo[0], hi[0], n_grid)

        for x0 in x0_vals:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break

            x = base_x.copy()
            x[0] = x0
            f = problem(x)
            evaluated.append((x, f))
            used += 1

        if used >= self.budget or time.time() > deadline:
            return [e[0] for e in evaluated]

        # Phase 2: Add some perturbation on x[1] to explore if it matters
        # and fine-tune around the sampled points
        remaining = self.budget - used

        # Try a few points with x[1] varied to check sensitivity
        # and add local refinements
        for i in range(min(remaining, 20)):
            if time.time() > deadline:
                break
            # Pick a random grid point and perturb x[1]
            idx = rng.randint(0, len(evaluated))
            x = evaluated[idx][0].copy()
            x[1] = lo[1] + rng.random() * (hi[1] - lo[1])
            f = problem(x)
            evaluated.append((x, f))
            used += 1

        # Phase 3: Fine-grained refinement around the entire front
        # Add intermediate points between grid points for denser coverage
        if used < self.budget and time.time() < deadline:
            # Create a finer grid with offset
            n_fine = min(self.budget - used, 100)
            if n_fine > 0:
                x0_fine = np.linspace(lo[0], hi[0], n_fine, endpoint=False)
                # Offset by half a grid step to fill gaps
                step = (hi[0] - lo[0]) / n_grid
                x0_fine = x0_fine + step / 2.0
                x0_fine = np.clip(x0_fine, lo[0], hi[0])

                for x0 in x0_fine:
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    x = base_x.copy()
                    x[0] = x0
                    f = problem(x)
                    evaluated.append((x, f))
                    used += 1

        return [e[0] for e in evaluated]
# EVOLVE-BLOCK-END
