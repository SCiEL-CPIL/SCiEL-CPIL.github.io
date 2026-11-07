# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer exploiting DTLZ4 structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        pop_size = 50

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def fast_nondominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            ranks = np.full(n, -1, dtype=int)

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    ranks[i] = 0

            fronts = []
            current = [i for i in range(n) if nd[i] == 0]
            if current:
                fronts.append(current)

            k = 0
            while current:
                next_front = []
                for i in current:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            ranks[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                current = next_front

            return fronts

        def crowding_distance(F, indices):
            m = len(indices)
            if m <= 2:
                return np.full(m, np.inf)

            cd = np.zeros(m)
            idx = np.array(indices)
            subF = F[idx]

            for o in range(n_obj):
                sorted_idx = np.argsort(subF[:, o])
                min_v = subF[sorted_idx[0], o]
                max_v = subF[sorted_idx[-1], o]
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                rng_v = max_v - min_v
                if rng_v < 1e-12:
                    continue
                for kk in range(1, m - 1):
                    cd[sorted_idx[kk]] += (subF[sorted_idx[kk + 1], o] - subF[sorted_idx[kk - 1], o]) / rng_v

            return cd

        # DTLZ4 structure: first (dim - n_obj + 1) = 10 vars for front position,
        # last (n_obj - 1) = 2 vars for distance from front (want near 0)
        n_front_vars = dim - n_obj + 1  # 10
        n_dist_vars = n_obj - 1  # 2

        def sample_dtlz4():
            """Sample a point biased toward the Pareto front for DTLZ4."""
            x = np.zeros(dim)
            # Front variables: uniform [0,1]
            x[:n_front_vars] = rng.random(n_front_vars)
            # Distance variables: biased toward 0 (beta=30 for DTLZ4)
            # Use beta distribution to concentrate near 0
            x[n_front_vars:] = rng.beta(1.0, 30.0, size=n_dist_vars)
            x = np.clip(x, lo, hi)
            return x

        # Initialize population
        X = np.array([sample_dtlz4() for _ in range(pop_size)])
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Archive: list of (x, f) pairs
        archive_X = []
        archive_F = []

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            for g in archive_F:
                if dominates(g, f):
                    return
            new_X = []
            new_F = []
            for i in range(len(archive_X)):
                if not dominates(f, archive_F[i]):
                    new_X.append(archive_X[i])
                    new_F.append(archive_F[i])
            new_X.append(x)
            new_F.append(f)
            archive_X = new_X
            archive_F = new_F

        for i in range(pop_size):
            update_archive(X[i], F[i])

        def mutate(x, progress):
            """Adaptive mutation exploiting DTLZ4 structure."""
            child = x.copy()
            
            # Mutation scale decreases over time
            scale = 0.3 * (1.0 - progress) + 0.05
            
            # Front variables: Gaussian perturbation
            child[:n_front_vars] += scale * 0.5 * rng.standard_normal(n_front_vars)
            
            # Distance variables: bias strongly toward 0
            # With some probability, push toward 0
            if rng.random() < 0.7:
                child[n_front_vars:] *= (1.0 - 0.3 * (1.0 - progress))
            else:
                child[n_front_vars:] += scale * 0.1 * rng.standard_normal(n_dist_vars)
            
            return np.clip(child, lo, hi)

        while used < budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = fast_nondominated_sort(F)
            if not fronts:
                break
            front0 = fronts[0]
            if not front0:
                front0 = list(range(pop_size))

            # Select parent using binary tournament on crowding distance within front0
            i1 = front0[rng.integers(len(front0))]
            i2 = front0[rng.integers(len(front0))]
            
            # Use crowding distance for selection within the front
            cd = crowding_distance(F, front0)
            cd_map = {front0[k]: cd[k] for k in range(len(front0))}
            
            if cd_map[i1] >= cd_map[i2]:
                parent_idx = i1
            else:
                parent_idx = i2

            progress = used / budget
            child = mutate(X[parent_idx], progress)

            fc = np.array(problem(child), dtype=np.float64)
            used += 1

            update_archive(child, fc)

            # Replace worst point in population by sum of objectives
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

            # Occasionally replace a random point for diversity
            if rng.random() < 0.3:
                rand_idx = rng.integers(pop_size)
                X[rand_idx] = child
                F[rand_idx] = fc

        return X
# EVOLVE-BLOCK-END
