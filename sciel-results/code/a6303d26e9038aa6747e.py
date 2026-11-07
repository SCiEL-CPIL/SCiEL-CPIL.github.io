"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using MOEA/D with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # MOEA/D parameters
        n_subproblems = 30
        T = 5  # neighborhood size
        max_iter = 15  # generations per subproblem update

        # Generate weight vectors uniformly
        weights = np.random.dirichlet(np.ones(self.n_obj), size=n_subproblems)

        # Find neighbors for each subproblem
        neighbors = []
        for i in range(n_subproblems):
            dists = np.linalg.norm(weights - weights[i], axis=1)
            neighbor_idx = np.argsort(dists)[:T + 1]
            neighbors.append(neighbor_idx)

        # Initialize population
        X = lo + range_ * rng.random((n_subproblems, self.dim))
        F = np.array([problem(x) for x in X])
        used = n_subproblems

        # Initialize ideal point
        ideal = np.min(F, axis=0)

        def normalize(F, ideal):
            """Normalize objectives using ideal point."""
            F_norm = F.copy()
            for j in range(self.n_obj):
                if ideal[j] < 1e-10:
                    F_norm[:, j] = F[:, j] / (F[:, j].max() + 1e-10)
                else:
                    F_norm[:, j] = (F[:, j] - ideal[j]) / (F[:, j].max() - ideal[j] + 1e-10)
            return F_norm

        def tchebycheff(F, weights, ideal):
            """Compute Tchebycheff aggregation values."""
            F_norm = normalize(F, ideal)
            return np.max(weights * F_norm, axis=1)

        def dominates(a, b):
            """Check if a dominates b (all objectives <= and at least one <)."""
            return np.all(a <= b) and np.any(a < b)

        # Main optimization loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            progress = used / self.budget
            mutation_scale = 0.2 * (1 - progress) + 0.03 * progress

            # Update each subproblem
            for i in range(n_subproblems):
                if used >= self.budget or time.time() > deadline:
                    break

                # Select parent from neighborhood
                neighbor_set = neighbors[i]
                # Tournament selection within neighborhood
                candidates = [neighbor_set[rng.integers(len(neighbor_set))] for _ in range(2)]
                parent_idx = min(candidates, key=lambda j: tchebycheff(F[j:j+1], weights[i:i+1], ideal)[0])

                # Crossover
                if rng.random() < 0.8:
                    child = np.zeros(self.dim)
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            child[d] = X[parent_idx, d]
                        else:
                            # Pick another parent from neighborhood
                            other_parent = neighbor_set[rng.integers(len(neighbor_set))]
                            child[d] = X[other_parent, d]
                else:
                    child = X[parent_idx].copy()

                # Mutation
                child += mutation_scale * range_ * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

                # Evaluate child
                child_F = np.array(problem(child))
                used += 1

                # Update ideal point
                ideal = np.minimum(ideal, child_F)

                # Update if child is better in Tchebycheff sense
                child_tch = tchebycheff(child_F[None, :], weights[i:i+1], ideal)[0]
                parent_tch = tchebycheff(F[parent_idx][None, :], weights[i:i+1], ideal)[0]

                if child_tch <= parent_tch:
                    X[i] = child
                    F[i] = child_F

                # Update neighbors if child dominates
                for j in neighbor_set:
                    if j != i and dominates(child_F, F[j]):
                        X[j] = child
                        F[j] = child_F

        return X
# EVOLVE-BLOCK-END
