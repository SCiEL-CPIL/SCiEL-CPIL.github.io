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

        # Population size: larger population for better coverage
        pop_size = min(100, max(30, budget // 2))
        if pop_size * 2 > budget:
            pop_size = max(10, budget // 2)

        # Generate weight vectors using simplex-lattice for 3 objectives
        weights = self._generate_weights(n_obj, pop_size, rng)

        # Ideal and nadir points for normalization (updated dynamically)
        ideal = np.full(n_obj, np.inf)
        nadir = np.full(n_obj, 0.0)

        # Initialize population with diverse points
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Update ideal/nadir
        ideal = np.min(F, axis=0)
        nadir = np.max(F, axis=0)
        # Avoid division by zero in normalization
        denom = nadir - ideal
        denom[denom < 1e-10] = 1.0

        # Neighborhood size for MOEA/D
        T = max(5, pop_size // 4)

        # Precompute neighbors
        neighbors = self._compute_neighbors(weights, T)

        # Adaptive sigma: start with larger mutation, decrease over time
        initial_sigma = 0.3
        final_sigma = 0.05
        total_iters = budget - pop_size

        while used < budget:
            # Select subproblem index - use round-robin with some randomness
            if rng.random() < 0.7:
                idx = rng.integers(pop_size)
            else:
                # Pick the worst subproblem (highest Tchebycheff value)
                fits = np.zeros(pop_size)
                for i in range(pop_size):
                    fn = (F[i] - ideal) / denom
                    fn = np.clip(fn, 0.0, 1.0)
                    fits[i] = np.max(weights[i] * fn)
                idx = np.argmax(fits)

            w = weights[idx]

            # Get neighbor indices
            nbrs = neighbors[idx]

            # Select parent from neighbors using fitness
            fitnesses = np.zeros(len(nbrs))
            for i, ni in enumerate(nbrs):
                f = F[ni]
                fn = (f - ideal) / denom
                fn = np.clip(fn, 0.0, 1.0)
                fitnesses[i] = np.max(w * fn)

            # Select parent with some probability based on fitness
            best_local = np.argmin(fitnesses)
            if rng.random() < 0.85:
                parent_idx = nbrs[best_local]
            else:
                parent_idx = nbrs[rng.integers(len(nbrs))]

            # Adaptive mutation sigma
            progress = (used - pop_size) / max(1, total_iters)
            sigma = initial_sigma * (1 - progress) + final_sigma * progress

            # Mutation: Gaussian perturbation with variable rate
            child = X[parent_idx].copy()
            mutation_rate = 0.3 + 0.4 * (1 - progress)  # higher mutation early
            mutation_mask = rng.random(dim) < mutation_rate
            if mutation_mask.sum() > 0:
                child[mutation_mask] += sigma * span[mutation_mask] * rng.standard_normal(mutation_mask.sum())
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Decide whether to replace X[idx]
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
            if np.any(fc < ideal):
                ideal = np.minimum(ideal, fc)
            if np.any(fc > nadir):
                nadir = np.maximum(nadir, fc)
            denom = nadir - ideal
            denom[denom < 1e-10] = 1.0

        # Return all evaluated points (X contains the final population)
        # We need to return all points that were evaluated, but the problem says
        # "Every returned objective vector is recorded" - so we just return X
        return X

    def _generate_weights(self, n_obj, pop_size, rng):
        """Generate weight vectors that sum to 1 and are non-negative."""
        weights = []
        # Use a grid-like approach for 3 objectives
        if n_obj == 3:
            # Generate points on the 2D simplex using stratified sampling
            # Use a triangular lattice
            n_per_edge = max(2, int(np.ceil(np.sqrt(pop_size * 2 / 3))))
            generated = []
            for i in range(n_per_edge + 1):
                for j in range(n_per_edge + 1 - i):
                    k = n_per_edge - i - j
                    w1 = i / n_per_edge
                    w2 = j / n_per_edge
                    w3 = k / n_per_edge
                    generated.append([w1, w2, w3])
            
            # Shuffle and take pop_size
            rng.shuffle(generated)
            if len(generated) >= pop_size:
                weights = generated[:pop_size]
            else:
                weights = generated
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
