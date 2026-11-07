"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast evolutionary multi-objective optimizer for WFG7 with front-seeded init, tournament selection, crossover, and elitist replacement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        rng = np.random.default_rng(42)

        # Population size
        pop_size = 30
        n_front_seed = int(0.6 * pop_size)  # 60% front-seeded
        n_random = pop_size - n_front_seed

        # Front-seeded initialization: structured points known to be near WFG7 Pareto front
        # For WFG7, we use a parameterized family. The front is on a curved surface.
        # We generate points by varying a parameter that controls the trade-off.
        front_X = np.zeros((n_front_seed, self.dim))
        for i in range(n_front_seed):
            # Parameter t in [0, 1] controls position along the front
            t = i / max(1, n_front_seed - 1)
            # For WFG7, a good parameterization involves distributing mass across objectives
            # We use a simple but effective heuristic: vary the bias across variables
            # This is a practical approximation - the exact front depends on the problem
            # Use a structured spread in decision space that empirically hits the front
            base = 0.5 + 0.4 * (2 * t - 1)  # Varies from 0.1 to 0.9
            # Add structured variation per variable
            for d in range(self.dim):
                # Each variable gets a slightly different offset to create diversity
                offset = 0.1 * np.sin(2 * np.pi * d / self.dim + t * np.pi)
                front_X[i, d] = lo[d] + (hi[d] - lo[d]) * np.clip(base + offset, 0.05, 0.95)

        # Random initialization for the rest
        random_X = lo + span * rng.random((n_random, self.dim))

        X = np.vstack([front_X, random_X])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: compute dominance matrix and non-dominated indices
        def get_non_dominated(F):
            N = F.shape[0]
            if N == 0:
                return np.array([], dtype=int), np.zeros(0, dtype=bool)
            F_j = F[:, np.newaxis, :]  # (N, 1, M)
            F_i = F[np.newaxis, :, :]  # (1, N, M)
            le = F_j <= F_i
            lt = F_j < F_i
            all_le = np.all(le, axis=2)
            any_lt = np.any(lt, axis=2)
            dom = all_le & any_lt
            np.fill_diagonal(dom, False)
            dominated = np.any(dom, axis=0)
            nd_indices = np.where(~dominated)[0]
            return nd_indices, dominated

        # Tournament selection: pick k random individuals, return the best (lowest sum)
        def tournament_select(X, F, k=2):
            N = len(F)
            indices = rng.integers(0, N, size=k)
            sums = np.sum(F[indices], axis=1)
            best_local = indices[np.argmin(sums)]
            return best_local

        while used < self.budget:
            # Time check
            if time.time() > deadline:
                break

            # Find non-dominated set
            nd_indices, dominated = get_non_dominated(F)

            if len(nd_indices) == 0:
                nd_indices = np.arange(pop_size)

            # Generate two children using tournament selection and crossover/mutation
            children_X = []
            children_F = []

            for _ in range(2):
                # Tournament selection: one parent biased toward best front point
                # With 50% probability, use the best non-dominated point as one parent
                if len(nd_indices) >= 2 and rng.random() < 0.5:
                    # Bias: one parent is the best (lowest sum) non-dominated point
                    nd_sums = np.sum(F[nd_indices], axis=1)
                    best_nd_idx = nd_indices[np.argmin(nd_sums)]
                    # Other parent is tournament-selected
                    other_idx = tournament_select(X, F, k=2)
                    parent1 = X[best_nd_idx]
                    parent2 = X[other_idx]
                elif len(nd_indices) >= 2:
                    # Both parents tournament-selected
                    p1_idx = tournament_select(X, F, k=2)
                    p2_idx = tournament_select(X, F, k=2)
                    parent1 = X[p1_idx]
                    parent2 = X[p2_idx]
                else:
                    # Only one non-dominated point, use it for both
                    parent1 = X[nd_indices[0]]
                    parent2 = X[nd_indices[0]]

                # Uniform crossover with mutation
                if rng.random() < 0.7:
                    # Crossover
                    mask = rng.random(self.dim) < 0.5
                    child = np.where(mask, parent1, parent2)
                else:
                    # Mutation only from parent1
                    child = parent1.copy()

                # Adaptive mutation
                progress = used / self.budget
                step_size = 0.12 * (1 - 0.5 * progress)
                child = child + step_size * span * rng.standard_normal(self.dim)

                child = np.clip(child, lo, hi)
                children_X.append(child)

            # Evaluate both children
            for child in children_X:
                fc = problem(child)
                used += 1
                children_F.append(fc)
                if used >= self.budget:
                    break

            if used >= self.budget:
                break

            # Elitist replacement: combine parents and children, keep best N
            # Combine current population and children
            all_X = np.vstack([X, np.array(children_X)])
            all_F = np.array([F, np.array(children_F)])
            all_N = len(all_F)

            # Compute dominance for all
            F_j = all_F[:, np.newaxis, :]
            F_i = all_F[np.newaxis, :, :]
            le = F_j <= F_i
            lt = F_j < F_i
            all_le = np.all(le, axis=2)
            any_lt = np.any(lt, axis=2)
            dom = all_le & any_lt
            np.fill_diagonal(dom, False)
            dominated_all = np.any(dom, axis=0)

            # Non-dominated indices in the combined set
            nd_all = np.where(~dominated_all)[0]

            if len(nd_all) >= pop_size:
                # Select pop_size from non-dominated using crowding distance
                # Simple crowding: use sum of objectives as a proxy
                nd_sums = np.sum(all_F[nd_all], axis=1)
                # Sort by sum, take best pop_size
                sorted_nd = nd_all[np.argsort(nd_sums)]
                selected = sorted_nd[:pop_size]
            else:
                # Include all non-dominated, fill rest with best dominated
                selected = list(nd_all)
                dom_all = np.where(dominated_all)[0]
                dom_sums = np.sum(all_F[dom_all], axis=1)
                sorted_dom = dom_all[np.argsort(dom_sums)]
                for idx in sorted_dom:
                    if len(selected) >= pop_size:
                        break
                    selected.append(idx)

            X = all_X[selected]
            F = all_F[selected]

        return X
# EVOLVE-BLOCK-END
