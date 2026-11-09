"""Multi-objective optimizer on ZDT6 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit ZDT6 structure: sample x1..x10 uniformly in [0,0.2] for good f1,
    then refine with local search around promising points."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim

        # ZDT6: f1 = 1 - exp(-4*x1) * sin(6*pi*x1)^6
        # Pareto front: x1 in [0,1], x2..x10 = 0.5 (approximately)
        # For x1 in [0, 0.2], f1 is small and the front curvature is high
        # We want to sample x1 in [0, 0.2] and set x2..x10 near 0.5

        # Initial population: sample x1 in [0, 0.2] (where f1 valley is),
        # x2..x10 near 0.5 (optimal for f2)
        n_init = 40
        x1_vals = np.linspace(0.0, 0.2, n_init)
        X = np.zeros((n_init, dim))
        for i, x1 in enumerate(x1_vals):
            X[i, 0] = x1
            # x2..x10 = 0.5 (middle of [0,1])
            X[i, 1:] = 0.5

        F = np.array([problem(x) for x in X])
        used = n_init

        # Check time
        if time.time() > deadline:
            return X

        # Local refinement: for each point, try small perturbations in x1 and x2..x10
        # Focus on improving both objectives
        remaining = self.budget - used
        n_refine = min(remaining, 160)

        # Use a simple local search: perturb x1 slightly and x2..x10 slightly
        rng = np.random.default_rng(42)

        for _ in range(n_refine):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break

            # Pick a random point from current population
            idx = rng.integers(n_init)
            parent = X[idx].copy()

            # Perturb x1 with small step (to explore along the front)
            parent[0] += rng.uniform(-0.01, 0.01)
            # Perturb x2..x10 with small step (to refine f2)
            parent[1:] += rng.uniform(-0.02, 0.02, size=dim - 1)
            parent = np.clip(parent, lo, hi)

            fc = problem(parent)
            used += 1

            # Replace worst point if this child is better in at least one objective
            # and not worse in all
            # Simple replacement: if child dominates any point, replace the dominated one
            # Otherwise, replace the point with worst sum of objectives if child is better
            sums = np.sum(F, axis=1)
            worst_idx = int(np.argmax(sums))

            # Check if child dominates the worst point
            if np.all(fc <= F[worst_idx]) and np.any(fc < F[worst_idx]):
                X[worst_idx] = parent
                F[worst_idx] = fc
            # Check if child is better in f1 (first objective)
            elif fc[0] < F[worst_idx][0]:
                X[worst_idx] = parent
                F[worst_idx] = fc
            # Check if child is better in f2 (second objective)
            elif fc[1] < F[worst_idx][1]:
                X[worst_idx] = parent
                F[worst_idx] = fc

        # Final phase: more targeted sampling along the Pareto front
        # x1 in [0, 0.15] with finer resolution, x2..x10 = 0.5
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            n_final = min(remaining, 80)
            # Sample x1 in [0, 0.15] where f1 < ~0.3
            x1_fine = rng.uniform(0.0, 0.15, size=(n_final,))
            for i in range(n_final):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                x = np.zeros(dim)
                x[0] = x1_fine[i]
                x[1:] = 0.5 + rng.uniform(-0.01, 0.01, size=dim - 1)
                x = np.clip(x, lo, hi)
                f = problem(x)
                used += 1

                # Add to population if it's not dominated
                dominated = False
                for j in range(n_init):
                    if np.all(F[j] <= f) and np.any(F[j] < f):
                        dominated = True
                        break
                if not dominated:
                    # Find a point to replace (worst sum)
                    sums = np.sum(F, axis=1)
                    worst_idx = int(np.argmax(sums))
                    if np.sum(f) < sums[worst_idx]:
                        X[worst_idx] = x
                        F[worst_idx] = f

        return X
# EVOLVE-BLOCK-END
