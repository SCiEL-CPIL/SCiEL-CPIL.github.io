"""Decomposition-based multi-objective optimizer for WFG9 (MOEA/D with Tchebycheff scalarization)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D with Tchebycheff scalarization, neighborhood mating, and external archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.perf_counter()
        deadline = t0 + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)

        # --- MOEA/D parameters ---
        N = 50  # population size
        T = 5   # neighborhood size
        CR = 0.7  # crossover rate (DE)
        F_mut = 0.8  # mutation factor
        F_cross = 0.5  # crossover probability

        # --- Generate weight vectors (uniform on simplex) ---
        weights = self._generate_weights(N, n_obj)

        # --- Neighborhood indices for each subproblem ---
        neighbors = self._compute_neighbors(weights, T)

        # --- Initialize population ---
        pop = lo + span * rng.random((N, dim))
        F = np.empty((N, n_obj))
        evals = 0
        all_F_list = []

        for i in range(N):
            F[i] = problem(pop[i])
            evals += 1
            all_F_list.append(F[i].copy())

        # --- External archive of non-dominated solutions ---
        archive_X = [pop[0].copy()]
        archive_F = [F[0].copy()]

        def _update_archive(x, f):
            """Add to archive if non-dominated by any archive member; remove dominated members."""
            non_dom = True
            new_archive_X = list(archive_X)
            new_archive_F = list(archive_F)

            for j in range(len(archive_F)):
                if np.all(archive_F[j] <= f) and np.any(archive_F[j] < f):
                    non_dom = False
                    break

            if non_dom:
                # Remove archive members dominated by new point
                keep = []
                for j in range(len(archive_F)):
                    if not (np.all(f <= archive_F[j]) and np.any(f < archive_F[j])):
                        keep.append(j)
                new_archive_X = [archive_X[j] for j in keep]
                new_archive_F = [archive_F[j] for j in keep]
                new_archive_X.append(x.copy())
                new_archive_F.append(f.copy())

                # Cap archive size
                if len(new_archive_X) > 200:
                    # Remove the point with worst crowding distance
                    cd = self._crowding_distance(np.array(new_archive_F))
                    worst = np.argmin(cd)
                    new_archive_X.pop(worst)
                    new_archive_F.pop(worst)

                archive_X[:] = new_archive_X
                archive_F[:] = new_archive_F

        # --- Tchebycheff scalarization ---
        def tchebycheff(f, w, z):
            """Compute Tchebycheff scalarization value."""
            return np.max(w * np.abs(f - z))

        # --- Ideal point (lower bound) ---
        z = F.min(axis=0).copy()

        # --- Main MOEA/D loop ---
        while evals < budget:
            if time.perf_counter() > deadline - 10:
                break

            # Select a random subproblem index
            m = rng.integers(N)

            # Select two parents from the neighborhood of m
            neigh = neighbors[m]
            p1_idx = neigh[rng.integers(len(neigh))]
            p2_idx = neigh[rng.integers(len(neigh))]

            # DE-style mutation and crossover
            y = pop[p1_idx] + F_mut * (pop[p1_idx] - pop[p2_idx])
            # Binary crossover
            cross_mask = rng.random(dim) < F_cross
            y = np.where(cross_mask, y, pop[p1_idx])
            # Mutation
            mut_idx = rng.integers(dim)
            y[mut_idx] += F_mut * (hi[mut_idx] - lo[mut_idx]) * (rng.random() - 0.5)
            y = np.clip(y, lo, hi)

            # Evaluate
            f_y = problem(y)
            evals += 1
            all_F_list.append(f_y.copy())

            # Update ideal point
            z = np.minimum(z, f_y)

            # Update archive
            _update_archive(y, f_y)

            # Update population using Tchebycheff scalarization
            # Replace the worst neighbor based on Tchebycheff value
            best_neighbor = -1
            best_val = np.inf
            for j in neigh:
                val = tchebycheff(F[j], weights[j], z)
                if val < best_val:
                    best_val = val
                    best_neighbor = j

            val_y = tchebycheff(f_y, weights[m], z)
            if val_y < best_val:
                pop[m] = y
                F[m] = f_y

        # Return the population
        return pop

    def _generate_weights(self, N, n_obj):
        """Generate N weight vectors uniformly on the (n_obj-1)-simplex."""
        if n_obj == 3:
            # Use a grid for 3 objectives
            # Generate points on the simplex with resolution
            k = int(np.ceil(np.sqrt(2 * N / 3)))
            weights = []
            for i in range(k + 1):
                for j in range(k + 1 - i):
                    w1 = i / k
                    w2 = j / k
                    w3 = 1.0 - w1 - w2
                    if w3 >= 0:
                        weights.append([w1, w2, w3])
            weights = np.array(weights)
            if len(weights) < N:
                # Add random points
                extra = np.random.default_rng(0).random((N - len(weights), n_obj))
                extra /= extra.sum(axis=1, keepdims=True)
                weights = np.vstack([weights, extra])
            elif len(weights) > N:
                # Randomly subsample
                idx = np.random.default_rng(0).choice(len(weights), N, replace=False)
                weights = weights[idx]
            return weights
        else:
            # Dirichlet distribution
            rng = np.random.default_rng(0)
            w = rng.dirichlet(np.ones(n_obj), size=N)
            return w

    def _compute_neighbors(self, weights, T):
        """Compute T nearest neighbors for each weight vector based on Euclidean distance."""
        N = len(weights)
        # Compute pairwise distances
        dists = np.zeros((N, N))
        for i in range(N):
            dists[i] = np.linalg.norm(weights - weights[i], axis=1)
        neighbors = []
        for i in range(N):
            order = np.argsort(dists[i])
            neighbors.append(order[1:T + 1].tolist())  # exclude self
        return neighbors

    def _crowding_distance(self, F_arr):
        """Compute crowding distance for a set of objective vectors."""
        n = len(F_arr)
        if n <= 2:
            return np.ones(n) * np.inf

        cd = np.zeros(n)
        for m in range(F_arr.shape[1]):
            sorted_idx = np.argsort(F_arr[:, m])
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf

            f_min = F_arr[sorted_idx[0], m]
            f_max = F_arr[sorted_idx[-1], m]
            if f_max - f_min < 1e-15:
                continue

            for k in range(1, n - 1):
                cd[sorted_idx[k]] += (F_arr[sorted_idx[k + 1], m] - F_arr[sorted_idx[k - 1], m]) / (f_max - f_min)

        return cd
# EVOLVE-BLOCK-END
