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

        # Population size: use a larger population for better coverage
        pop_size = min(100, max(30, budget // 2))
        if pop_size * 2 > budget:
            pop_size = max(15, budget // 2)

        # Generate well-distributed weight vectors on the simplex
        weights = self._generate_weights(n_obj, pop_size, rng)

        # Ideal and nadir points for normalization
        ideal = np.full(n_obj, np.inf)
        nadir = np.full(n_obj, 0.0)

        # Initialize population with diverse random points
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Update ideal/nadir
        ideal = np.min(F, axis=0)
        nadir = np.max(F, axis=0)
        denom = nadir - ideal
        denom[denom < 1e-10] = 1.0

        # Track all evaluated points for hypervolume calculation
        all_F = [F.copy()]
        all_X = [X.copy()]

        # Neighborhood size for MOEA/D - larger neighborhood for better diversity
        T = max(5, pop_size // 4)

        # Precompute neighbors
        neighbors = self._compute_neighbors(weights, T)

        # Adaptive mutation rate that decreases over time
        initial_sigma = 0.2
        final_sigma = 0.02
        total_iterations = budget - pop_size
        iteration = 0

        while used < budget:
            # Select subproblem index - cycle through weights for systematic coverage
            idx = iteration % pop_size
            w = weights[idx]

            # Get neighbor indices
            nbrs = neighbors[idx]

            # Compute Tchebycheff fitness for each neighbor
            fitnesses = np.zeros(len(nbrs))
            for i, ni in enumerate(nbrs):
                f = F[ni]
                fn = (f - ideal) / denom
                fn = np.clip(fn, 0.0, 1.0)
                fitnesses[i] = np.max(w * fn)

            # Select parent: best with high probability, random with low probability
            best_local = np.argmin(fitnesses)
            if rng.random() < 0.9:
                parent_idx = nbrs[best_local]
            else:
                parent_idx = nbrs[rng.integers(len(nbrs))]

            # Adaptive mutation: sigma decreases over time
            progress = iteration / max(1, total_iterations)
            sigma = initial_sigma + (final_sigma - initial_sigma) * progress
            
            # Polynomial mutation for better exploration
            child = X[parent_idx].copy()
            mutation_mask = rng.random(dim) < 0.6
            if mutation_mask.sum() > 0:
                eta = 20.0
                u = rng.random(mutation_mask.sum())
                if u < 0.5:
                    factor = (2 * u) ** (1.0 / (eta + 1))
                else:
                    factor = (2 * (1 - u)) ** (1.0 / (eta + 1))
                factor = 1 + (2 * (factor - 1)) * rng.random(mutation_mask.sum())
                child[mutation_mask] += sigma * span[mutation_mask] * factor
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1
            iteration += 1

            # Store all evaluated points
            all_F.append(np.array([fc]))
            all_X.append(np.array([child]))

            # Decide whether to replace X[idx] using Tchebycheff fitness
            fn_child = (fc - ideal) / denom
            fn_child = np.clip(fn_child, 0.0, 1.0)
            fit_child = np.max(w * fn_child)

            fn_curr = (F[idx] - ideal) / denom
            fn_curr = np.clip(fn_curr, 0.0, 1.0)
            fit_curr = np.max(w * fn_curr)

            # Also check if child dominates current solution
            dominates = np.all(fc < F[idx])
            
            if fit_child < fit_curr or dominates:
                X[idx] = child
                F[idx] = fc

            # Update ideal and nadir
            ideal = np.minimum(ideal, fc)
            nadir = np.maximum(nadir, fc)
            denom = nadir - ideal
            denom[denom < 1e-10] = 1.0

        # Combine all evaluated points
        combined_F = np.vstack(all_F)
        combined_X = np.vstack(all_X)

        return combined_X

    def _generate_weights(self, n_obj, pop_size, rng):
        """Generate well-distributed weight vectors on the simplex."""
        if n_obj == 3:
            # Use a systematic grid on the 2D simplex for uniform coverage
            weights = []
            # Find a good resolution for the simplex
            # For 3 objectives, we need i+j+k = n where weights = [i/n, j/n, k/n]
            # Number of points = (n+1)(n+2)/2
            # We want approximately pop_size points
            # Solve (n+1)(n+2)/2 ≈ pop_size
            n = int((-3 + np.sqrt(9 + 8 * pop_size)) / 2)
            n = max(1, n)
            
            for i in range(n + 1):
                for j in range(n + 1 - i):
                    k = n - i - j
                    w = np.array([i, j, k], dtype=float) / n
                    weights.append(w)
            
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
            dists[i] = np.inf
            nbr_indices = np.argsort(dists)[:T]
            neighbors.append(nbr_indices.tolist())
        return neighbors
# EVOLVE-BLOCK-END
