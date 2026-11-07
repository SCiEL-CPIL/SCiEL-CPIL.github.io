"""Decomposition-based multi-objective optimizer for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Decomposition-based MOEA/D with Tchebycheff scalarization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Population size: balance exploration and exploitation
        N = min(80, max(20, self.budget // 3))
        N = min(N, 100)  # cap for safety

        # Generate reference points for Tchebycheff scalarization
        # Use a grid of weight vectors in (n_obj - 1) dimensions
        # For 3 objectives, we generate weights that sum to 1
        ref_points = self._generate_ref_points(n_obj, N, rng)

        # Initialize population with random points
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        # Initialize ideal point (best found for each objective)
        ideal = np.min(F, axis=0)

        # Track best hypervolume-contributing points
        all_F = [F.copy()]  # store snapshots for final hypervolume

        # Neighborhood size
        T = max(2, N // 5)

        # Build neighborhood indices
        # Sort by first weight component for locality
        order = np.argsort(ref_points[:, 0])
        neighbors = [list(range(max(0, i - T // 2), min(N, i + T // 2 + 1))) for i in range(N)]
        # Shuffle to avoid bias
        for i in range(N):
            rng.shuffle(neighbors[i])

        # Main evolutionary loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Update ideal point
            ideal = np.min(ideal, np.min(F, axis=0))

            # For each subproblem, select a parent from its neighborhood
            for i in range(N):
                if used >= self.budget or time.time() > deadline:
                    break

                # Select parent from neighborhood of i
                if len(neighbors[i]) > 0:
                    parent_idx = neighbors[i][rng.integers(len(neighbors[i]))]
                else:
                    parent_idx = rng.integers(N)

                parent = X[parent_idx].copy()

                # Mutation: Gaussian perturbation
                # Adaptive step size
                step = 0.15 * (hi - lo)
                child = parent + step * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                # Evaluate
                fc = problem(child)
                used += 1

                # Update ideal
                ideal = np.min(ideal, fc)

                # Replacement: accept if child is better in its subproblem's scalarization
                # or replace the worst neighbor
                best_neighbor_idx = i
                best_scalar = self._tchebycheff(F[i], ref_points[i], ideal)

                for j in neighbors[i]:
                    if j == i:
                        continue
                    # Check if child is better than j in j's subproblem
                    child_scalar_j = self._tchebycheff(fc, ref_points[j], ideal)
                    current_scalar_j = self._tchebycheff(F[j], ref_points[j], ideal)
                    if child_scalar_j < current_scalar_j:
                        best_neighbor_idx = j
                        best_scalar = child_scalar_j
                        break

                # Replace
                X[best_neighbor_idx] = child
                F[best_neighbor_idx] = fc

        # Final: collect all non-dominated points for hypervolume
        # Return the final population
        return X

    def _generate_ref_points(self, n_obj, N, rng):
        """Generate N weight vectors in [0,1] that sum to 1."""
        points = []
        for _ in range(N):
            # Dirichlet distribution for diverse weights
            w = rng.dirichlet(np.ones(n_obj))
            points.append(w)
        return np.array(points)

    def _tchebycheff(self, f, w, ideal):
        """Tchebycheff scalarization value."""
        # Handle case where ideal might be >= f (shouldn't happen normally)
        diff = f - ideal
        # Use max to avoid division by zero issues
        return np.max(w * diff / np.maximum(w, 1e-10))
# EVOLVE-BLOCK-END
