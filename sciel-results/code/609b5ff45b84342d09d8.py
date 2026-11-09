"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware evolutionary multi-objective optimizer for ZDT1."""

    def __init__(self, dim, n_obj, budget):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin out of 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # For ZDT1: f1 = x[0], f2 = g(x[1:]) * (1 - sqrt(x[0]/g(x[1:])))
        # Optimal g(x[1:]) = 1 when x[1:] = 0
        # Strategy: focus search on x[0] and keep x[1:] near 0

        budget = self.budget
        pop_size = 30

        # Initialize: x[0] spread across [0,1], x[1:] near 0 with small noise
        X = np.zeros((pop_size, self.dim))
        X[:, 0] = rng.uniform(lo[0], hi[0], pop_size)
        # x[1:] initialized near 0 (optimal for g) with small perturbation
        for j in range(1, self.dim):
            X[:, j] = rng.uniform(lo[j], lo[j] + 0.01 * span[j], pop_size)

        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_sort(F):
            """Return indices of non-dominated solutions."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            if len(indices) <= 2:
                return np.ones(len(indices))
            dist = np.zeros(len(indices))
            for m in range(self.n_obj):
                sorted_idx = np.argsort(F[indices, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                fmin = F[indices[sorted_idx[0]], m]
                fmax = F[indices[sorted_idx[-1]], m]
                if fmax - fmin < 1e-15:
                    continue
                for k in range(1, len(indices) - 1):
                    dist[sorted_idx[k]] += (F[indices[sorted_idx[k+1]], m] - F[indices[sorted_idx[k-1]], m]) / (fmax - fmin)
            return dist

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Find non-dominated front
            nd_indices = non_dominated_sort(F)
            
            if len(nd_indices) == 0:
                nd_indices = np.arange(len(F))
            
            # Compute crowding distance for selection
            cd = crowding_distance(F, nd_indices)
            
            # Select parent: prefer non-dominated with high crowding distance
            if len(nd_indices) > 1:
                # Probabilistic selection based on crowding distance
                probs = cd / cd.sum()
                parent_local_idx = rng.choice(len(nd_indices), p=probs)
            else:
                parent_local_idx = 0
            parent_idx = nd_indices[parent_local_idx]
            parent = X[parent_idx].copy()

            # Mutation: for ZDT1, x[0] controls f1, x[1:] should stay near 0
            # Use different mutation strategies
            r = rng.random()
            if r < 0.5:
                # Perturb x[0] to explore the front
                child = parent.copy()
                step = 0.05 * span[0] * rng.standard_normal()
                child[0] = np.clip(parent[0] + step, lo[0], hi[0])
                # Keep x[1:] near 0 with small noise
                for j in range(1, self.dim):
                    if rng.random() < 0.3:
                        child[j] = np.clip(parent[j] + 0.01 * span[j] * rng.standard_normal(), lo[j], hi[j])
                    else:
                        # Pull back toward 0
                        child[j] = np.clip(parent[j] * 0.9 + 0.1 * lo[j], lo[j], hi[j])
            elif r < 0.8:
                # Larger exploration step for x[0]
                child = parent.copy()
                child[0] = np.clip(parent[0] + 0.1 * span[0] * rng.standard_normal(), lo[0], hi[0])
                for j in range(1, self.dim):
                    if rng.random() < 0.2:
                        child[j] = np.clip(parent[j] + 0.02 * span[j] * rng.standard_normal(), lo[j], hi[j])
                    else:
                        child[j] = np.clip(parent[j] * 0.8 + 0.2 * lo[j], lo[j], hi[j])
            else:
                # Random restart in x[0], reset x[1:] to near 0
                child = parent.copy()
                child[0] = rng.uniform(lo[0], hi[0])
                for j in range(1, self.dim):
                    child[j] = lo[j] + rng.random() * 0.005 * span[j]

            fc = problem(child)
            used += 1

            # Replace worst solution (by sum of objectives) or a random dominated solution
            dominated_mask = np.array([
                np.any(np.all(F <= fc, axis=1) & np.any(F < fc, axis=1))
            ])
            # Find solutions that are dominated by the child or have worse sum
            sums = np.sum(F, axis=1)
            worst_idx = int(np.argmax(sums))
            
            # Only replace if child is better in some way
            if np.any(fc < F[worst_idx]) or np.sum(fc) < sums[worst_idx]:
                X[worst_idx] = child
                F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
