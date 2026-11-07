"""Decomposition-based MOEA/D for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """MOEA/D-style decomposition multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: keep it reasonable to allow multiple generations
        pop_size = min(60, max(20, budget // 3))
        if pop_size * 2 > budget:
            pop_size = max(10, budget // 2)

        # Generate weight vectors using simplex-lattice or random normalized weights
        # For 3 objectives, use a set of weight vectors that cover the simplex
        weights = self._generate_weights(n_obj, pop_size, rng)

        # Ideal and nadir points for normalization (updated dynamically)
        ideal = np.full(n_obj, np.inf)
        nadir = np.full(n_obj, 0.0)

        # Initialize population
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Update ideal/nadir
        ideal = np.min(F, axis=0)
        nadir = np.max(F, axis=0)
        # Avoid division by zero in normalization
        denom = nadir - ideal
        denom[denom < 1e-10] = 1.0

        # Track best solutions found so far (archive)
        archive_F = F.copy()
        archive_X = X.copy()

        # Neighborhood size for MOEA/D
        T = max(2, pop_size // 5)

        # Precompute neighbors
        neighbors = self._compute_neighbors(weights, T)

        while used < budget:
            # Select subproblem index
            idx = rng.integers(pop_size)
            w = weights[idx]

            # Get neighbor indices
            nbrs = neighbors[idx]

            # Select parent from neighbors using fitness
            # Compute Tchebycheff fitness for each neighbor
            fitnesses = np.zeros(len(nbrs))
            for i, ni in enumerate(nbrs):
                f = F[ni]
                # Normalized objectives
                fn = (f - ideal) / denom
                fn = np.clip(fn, 0.0, 1.0)
                # Tchebycheff aggregation
                fitnesses[i] = np.max(w * fn)

            # Select parent with some probability based on fitness
            # Use tournament-like selection: pick best with some randomness
            best_local = np.argmin(fitnesses)
            if rng.random() < 0.8:
                parent_idx = nbrs[best_local]
            else:
                parent_idx = nbrs[rng.integers(len(nbrs))]

            # Mutation: Gaussian perturbation
            sigma = 0.1
            child = X[parent_idx].copy()
            mutation_mask = rng.random(dim) < 0.5
            child[mutation_mask] += sigma * span[mutation_mask] * rng.standard_normal(mutation_mask.sum())
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Update archive
            archive_F = np.vstack([archive_F, [fc]])
            archive_X = np.vstack([archive_X, [child]])

            # Decide whether to replace X[idx]
            # Compare child with current solution at idx using Tchebycheff
            fn_child = (fc - ideal) / denom
            fn_child = np.clip(fn_child, 0.0, 1.0)
            fit_child = np.max(w * fn_child)

            fn_curr = (F[idx] - ideal) / denom
            fn_curr = np.clip(fn_curr, 0.0, 1.0)
            fit_curr = np.max(w * fn_curr)

            if fit_child <= fit_curr:
                X[idx] = child
                F[idx] = fc

            # Update ideal and nadir
            ideal = np.min(np.vstack([ideal, fc]), axis=0)
            nadir = np.max(np.vstack([nadir, fc]), axis=0)
            denom = nadir - ideal
            denom[denom < 1e-10] = 1.0

        # Return the archive (all evaluated points)
        return archive_X

    def _generate_weights(self, n_obj, pop_size, rng):
        """Generate weight vectors that sum to 1 and are non-negative."""
        weights = []
        # Use a grid-like approach for 3 objectives
        if n_obj == 3:
            # Generate points on the 2D simplex
            # Use stratified sampling
            n_per_edge = max(1, int(np.sqrt(pop_size / 3)))
            for i in range(n_per_edge + 1):
                for j in range(n_per_edge + 1 - i):
                    k = n_per_edge - i - j
                    w1 = i / n_per_edge
                    w2 = j / n_per_edge
                    w3 = k / n_per_edge
                    weights.append([w1, w2, w3])
            # If we have too many, subsample
            if len(weights) > pop_size:
                indices = rng.choice(len(weights), pop_size, replace=False)
                weights = [weights[i] for i in indices]
            # If too few, add random weights
            while len(weights) < pop_size:
                w = rng.dirichlet(np.ones(n_obj))
                weights.append(w)
        else:
            weights = [rng.dirichlet(np.ones(n_obj)) for _ in range(pop_size)]

        weights = np.array(weights, dtype=float)
        # Normalize to sum to 1
        sums = weights.sum(axis=1, keepdims=True)
        weights = weights / sums
        return weights

    def _compute_neighbors(self, weights, T):
        """Compute T nearest neighbors for each weight vector based on Euclidean distance."""
        n = len(weights)
        neighbors = []
        for i in range(n):
            dists = np.sum((weights - weights[i]) ** 2, axis=1)
            # Exclude self
            dists[i] = np.inf
            nbr_indices = np.argsort(dists)[:T]
            neighbors.append(nbr_indices.tolist())
        return neighbors
# EVOLVE-BLOCK-END
