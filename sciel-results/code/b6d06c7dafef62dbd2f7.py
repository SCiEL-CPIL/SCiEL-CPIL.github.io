"""Multi-objective optimizer on WFG9 using MOEA/D."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D multi-objective evolutionary optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 300s limit, stop at 250s
        deadline = start_time + 250

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # 1. Generate Weight Vectors on a Grid
        # For 3 objectives, use a grid with K steps
        K = 5
        weights = []
        for i in range(K + 1):
            for j in range(K + 1 - i):
                k = K - i - j
                w = np.array([i, j, k], dtype=float)
                if np.sum(w) > 0:
                    w /= np.sum(w)
                else:
                    w = np.array([1/3, 1/3, 1/3])
                weights.append(w)
        weights = np.array(weights)
        pop_size = len(weights)

        # 2. Initialize Population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Ideal point (min values seen so far)
        z_star = F.min(axis=0)

        # 3. Pre-compute Neighbors
        # Find 5 nearest neighbors for each weight vector
        neighbor_list = []
        for i in range(pop_size):
            dists = np.linalg.norm(weights - weights[i], axis=1)
            dists[i] = np.inf
            neighbors = np.argsort(dists)[:5]
            neighbor_list.append(neighbors)

        # Aggregation function: Tchebycheff
        def aggregate(f, w):
            return np.max(w * np.abs(f - z_star))

        # Scalar fitness for each individual
        G = np.array([aggregate(f, w) for f, w in zip(F, weights)])

        # 4. Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Select a weight vector to update
            i = rng.integers(pop_size)

            # Select parents from neighbors of i
            neigh = neighbor_list[i]
            # Select 2 parents from neighbors + self
            candidate_parents = np.concatenate(([i], neigh))
            parent_indices = rng.choice(len(candidate_parents), 2, replace=False)
            p1_idx = candidate_parents[parent_indices[0]]
            p2_idx = candidate_parents[parent_indices[1]]

            p1 = X[p1_idx]
            p2 = X[p2_idx]

            # Crossover: SBX (Simulated Binary Crossover)
            # eta_c = 20
            eta_c = 20
            child = np.zeros(self.dim)
            for d in range(self.dim):
                if rng.random() < 0.5:
                    if abs(p1[d] - p2[d]) > 1e-14:
                        if rng.random() < 0.5:
                            beta = (2 * rng.random()) ** (1 / (eta_c + 1))
                        else:
                            beta = (1 / (2 * (1 - rng.random()))) ** (1 / (eta_c + 1))
                        child[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
                    else:
                        child[d] = p1[d]
                else:
                    if abs(p1[d] - p2[d]) > 1e-14:
                        if rng.random() < 0.5:
                            beta = (2 * (1 - rng.random())) ** (1 / (eta_c + 1))
                        else:
                            beta = (1 / (2 * rng.random())) ** (1 / (eta_c + 1))
                        child[d] = 0.5 * ((1 + beta) * p2[d] + (1 - beta) * p1[d])
                    else:
                        child[d] = p2[d]

            # Mutation: Polynomial Mutation
            eta_m = 20
            for d in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    x_l = lo[d]
                    x_u = hi[d]
                    if abs(x_u - x_l) < 1e-14:
                        continue
                    if rng.random() < 0.5:
                        delta1 = (child[d] - x_l) / (x_u - x_l)
                        deltaq = (1 + 2 * delta1) ** (1 / (eta_m + 1))
                        if rng.random() < 0.5:
                            xk = 1 - deltaq
                        else:
                            xk = 1
                        delta = (xk ** (eta_m + 1) - 1)
                        mut = x_l + delta * (x_u - x_l) / 2
                    else:
                        delta2 = (x_u - child[d]) / (x_u - x_l)
                        deltaq = (1 + 2 * delta2) ** (1 / (eta_m + 1))
                        if rng.random() < 0.5:
                            xk = 1 - deltaq
                        else:
                            xk = 1
                        delta = 1 - (xk ** (eta_m + 1))
                        mut = x_u - delta * (x_u - x_l) / 2
                    child[d] = mut

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Update Ideal Point
            z_star = np.minimum(z_star, fc)

            # Calculate scalar fitness of child
            g_child_i = aggregate(fc, weights[i])

            # Update self (i)
            if g_child_i < G[i]:
                X[i] = child
                F[i] = fc
                G[i] = g_child_i

            # Update neighbors
            for j in neighbor_list[i]:
                g_child_j = aggregate(fc, weights[j])
                if g_child_j < G[j]:
                    X[j] = child
                    F[j] = fc
                    G[j] = g_child_j

        # 5. Return Non-Dominated Set
        non_dom_mask = np.ones(len(F), dtype=bool)
        for i in range(len(F)):
            if not non_dom_mask[i]:
                continue
            for j in range(len(F)):
                if i == j or not non_dom_mask[j]:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    non_dom_mask[j] = False
        return X[non_dom_mask]
# EVOLVE-BLOCK-END
