"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


class Optimizer:
    """MOEA/D: Multi-objective optimization via decomposition with Tchebycheff aggregation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _generate_weights(self, n_obj, n_sub):
        """Generate weight vectors on the simplex for n_obj objectives."""
        weights = []
        if n_obj == 1:
            weights.append(np.array([1.0]))
            return np.array(weights[:n_sub]) if n_sub >= 1 else np.array([[1.0]])
        if n_obj == 2:
            for i in range(n_sub):
                w1 = i / max(n_sub - 1, 1)
                weights.append(np.array([w1, 1.0 - w1]))
            return np.array(weights)
        if n_obj == 3:
            # Generate points on the 2D simplex for 3 objectives
            # Use systematic grid
            points = []
            # We need to generate n_sub points on the triangle
            # Use a spiral or grid approach
            # Simple approach: use all combinations with sufficient resolution
            # For 3 objectives, we can use: for i in range(n+1): for j in range(n+1-i):
            # w = [i/n, j/n, (n-i-j)/n]
            # But we need exactly n_sub points. Let's use a different approach.
            # Generate points using a 2D grid on the triangle
            # Alternative: use scipy's Delaunay or just use a spiral
            # Simplest: use the formula with a parameter
            # Let's use a systematic approach
            n_grid = int(np.ceil(np.sqrt(n_sub)))
            while True:
                points = []
                for i in range(n_grid + 1):
                    for j in range(n_grid + 1 - i):
                        w1 = i / n_grid
                        w2 = j / n_grid
                        w3 = 1.0 - w1 - w2
                        if w3 >= -1e-10:
                            points.append([max(w1, 0), max(w2, 0), max(w3, 0)])
                if len(points) >= n_sub:
                    break
                n_grid += 1
            # Normalize and select n_sub points
            points = np.array(points)
            # If we have more, subsample uniformly
            if len(points) > n_sub:
                idx = np.linspace(0, len(points) - 1, n_sub, dtype=int)
                points = points[idx]
            # If we have less, pad by repeating
            while len(points) < n_sub:
                points = np.vstack([points, points[0:1]])
            return points[:n_sub]
        # Fallback: random weights
        W = np.random.dirichlet(np.ones(n_obj), size=n_sub)
        return W

    def _tchebycheff(self, f, w, z):
        """Tchebycheff aggregation: max_i w_i * (f_i - z_i) / z_range_i"""
        # z is the ideal point (minimum of each objective)
        # We need to handle the range. Use a large number or compute dynamically.
        # Standard MOEA/D uses: g = max_i w_i * |f_i - z_i| / z_range_i
        # But z_range needs to be known. We'll use a dynamic approach.
        # Actually, for simplicity and robustness, use:
        # g = max_i w_i * (f_i - z_i) where z_i is the current best (minimum) for obj i
        # This is a simplified Tchebycheff that works well in practice.
        diff = f - z
        # Avoid negative values by clipping (since z is the minimum, f_i - z_i >= 0)
        diff = np.maximum(diff, 0)
        return np.max(w * diff)

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Number of subproblems
        n_sub = min(budget, 30)  # Keep it manageable
        n_sub = max(n_sub, 5)

        # Generate weight vectors
        W = self._generate_weights(n_obj, n_sub)

        # Initialize population
        X = lo + (hi - lo) * rng.random((n_sub, dim))
        F = np.array([problem(x) for x in X])
        used = n_sub

        # Initialize ideal point (minimum of each objective)
        z = np.min(F, axis=0)

        # Find neighbors for each subproblem (based on weight vector distance)
        # Compute pairwise distances between weight vectors
        dists = np.zeros((n_sub, n_sub))
        for i in range(n_sub):
            for j in range(n_sub):
                dists[i, j] = np.linalg.norm(W[i] - W[j])
        # For each subproblem, find the closest n_neighbors
        n_neighbors = min(20, n_sub)
        neighbors = []
        for i in range(n_sub):
            order = np.argsort(dists[i])
            neighbors.append(order[1:n_neighbors + 1])  # Exclude self

        # Main loop
        while used < budget:
            # Select a subproblem to optimize
            # Use random selection with some bias
            idx = rng.integers(n_sub)

            # Select parent from neighbors of idx
            nb = neighbors[idx]
            parent_idx = nb[rng.integers(len(nb))]

            # Crossover: SBX or simple blend
            # Use blend crossover
            alpha = rng.random()
            child = alpha * X[parent_idx] + (1 - alpha) * X[idx]
            # Add mutation
            if rng.random() < 0.5:
                child = child + 0.1 * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            # Evaluate
            fc = problem(child)
            used += 1

            # Update ideal point
            z = np.minimum(z, fc)

            # Update the solution for subproblem idx if it improves the Tchebycheff value
            g_old = self._tchebycheff(F[idx], W[idx], z)
            g_new = self._tchebycheff(fc, W[idx], z)
            if g_new < g_old:
                X[idx] = child
                F[idx] = fc

            # Update neighboring subproblems
            for nb_idx in neighbors[idx]:
                g_nb_old = self._tchebycheff(F[nb_idx], W[nb_idx], z)
                g_nb_new = self._tchebycheff(fc, W[nb_idx], z)
                if g_nb_new < g_nb_old:
                    X[nb_idx] = child
                    F[nb_idx] = fc

        return X
# EVOLVE-BLOCK-END
