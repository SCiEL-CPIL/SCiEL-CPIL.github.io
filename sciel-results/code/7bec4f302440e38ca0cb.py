"""Rewritten multi-objective optimizer for WFG4 using a decomposition-based MOEA/D approach."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """MOEA/D-style multi-objective optimizer with adaptive neighborhood search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Generate weight vectors using simplex lattice
        n_sub = 20  # number of subproblems
        weights = self._generate_weights(n_sub, n_obj)
        # Normalize weights
        norms = np.linalg.norm(weights, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        weights = weights / norms

        # Initialize population with random points
        pop_size = n_sub
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Track best solutions found
        best_points = []
        for i in range(pop_size):
            best_points.append((F[i].copy(), X[i].copy()))

        # Adaptive mutation parameters
        sigma = 0.1  # initial mutation strength
        sigma_decay = 0.995

        # Neighborhood size
        T = min(5, pop_size)

        # Main optimization loop
        while used < budget:
            # Select a random subproblem
            idx = rng.integers(pop_size)
            w = weights[idx]

            # Get neighborhood indices
            neighbors = self._get_neighbors(idx, pop_size, T, weights, rng)

            # Select parent from neighborhood using Tchebycheff fitness
            best_parent_idx = idx
            best_fitness = np.inf
            for n_idx in neighbors:
                # Tchebycheff fitness: max_i(w_i * |f_i - z_i|)
                # Use current best objective values as ideal point approximation
                fitness = np.max(w * np.abs(F[n_idx]))
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_parent_idx = n_idx

            # Mutate the selected parent
            parent = X[best_parent_idx]
            # Adaptive mutation with decreasing sigma
            mutation = rng.standard_normal(dim)
            child = parent + sigma * (hi - lo) * mutation
            child = np.clip(child, lo, hi)

            # Evaluate child
            fc = np.array(problem(child), dtype=float)
            used += 1
            sigma *= sigma_decay

            # Update the subproblem if child is better in Tchebycheff sense
            child_fitness = np.max(w * np.abs(fc))
            if child_fitness < F[idx] @ w:
                X[idx] = child
                F[idx] = fc

            # Also try to replace in neighborhood if dominated
            for n_idx in neighbors:
                if n_idx == idx:
                    continue
                # Check if child dominates existing solution
                if np.all(fc <= F[n_idx]) and np.any(fc < F[n_idx]):
                    X[n_idx] = child
                    F[n_idx] = fc

            # Update best points list
            best_points.append((fc.copy(), child.copy()))

            # Periodic diversity injection
            if used % 50 == 0 and used < budget:
                # Replace one random individual with a new random point
                ridx = rng.integers(pop_size)
                new_x = lo + (hi - lo) * rng.random(dim)
                new_f = np.array(problem(new_x), dtype=float)
                used += 1
                X[ridx] = new_x
                F[ridx] = new_f
                best_points.append((new_f.copy(), new_x.copy()))

        # Final cleanup: return the population
        return X

    def _generate_weights(self, n_sub, n_obj):
        """Generate weight vectors using simplex lattice method."""
        weights = []
        # Use recursive generation for simplex lattice
        if n_obj == 1:
            weights.append([1.0])
        else:
            # Generate all combinations of n_sub-1 points summing to 1
            # For simplicity, use random generation with normalization
            for _ in range(n_sub):
                w = rng_simplex(n_obj)
                weights.append(w)
        return np.array(weights)

    def _get_neighbors(self, idx, pop_size, T, weights, rng):
        """Get T nearest neighbors based on weight vector distance."""
        if pop_size <= T:
            return list(range(pop_size))
        
        w_idx = weights[idx]
        distances = np.linalg.norm(weights - w_idx, axis=1)
        # Exclude self
        distances[idx] = np.inf
        neighbor_indices = np.argsort(distances)[:T]
        return list(neighbor_indices)


def rng_simplex(n_obj):
    """Generate a random point in the (n_obj-1)-dimensional simplex."""
    # Use Dirichlet distribution
    a = np.ones(n_obj)
    point = np.random.dirichlet(a)
    return point
# EVOLVE-BLOCK-END
