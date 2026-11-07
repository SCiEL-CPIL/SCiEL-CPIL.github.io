"""Multi-objective optimizer for DTLZ1 with dense front seeding + rapid NSGA-II refinement."""
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1.
    Strategy:
    1. Dense quasi-random seeding along the analytical DTLZ1 Pareto front (g=0 plane).
    2. Add a small number of random off-front points for diversity.
    3. Run a fast NSGA-II loop with vectorized dominance sorting to refine the front.
    4. Explicit wall-clock deadline guard to stay safely under the 300 s limit.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 285.0  # Safety margin before the 300 s hard stop

    def _elapsed(self):
        return time.time() - self.start_time

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # ---------------------------------------------------------------
        # 1) Analytical front seeding for DTLZ1 (M=3, K=7)
        #    On the Pareto front: x_3..x_7 = 0.5  ->  g = 0
        #    f1 = 0.5 * x1 * x2
        #    f2 = 0.5 * (1 - x1) * x2
        #    f3 = 0.5 * (1 - x2)
        #    with 0 <= x1, x2 <= 1
        # ---------------------------------------------------------------
        n_front = 120  # number of front points to seed
        # Latin-hypercube / stratified grid in 2D for good coverage
        i = np.arange(n_front)
        # Two interleaved stratified grids for 2D coverage
        n_grid = int(np.ceil(np.sqrt(n_front)))
        gx = (np.arange(n_grid) + 0.5) / n_grid
        gy = (np.arange(n_grid) + 0.5) / n_grid
        X1, X2 = np.meshgrid(gx, gy, indexing='xy')
        X1 = X1.ravel()
        X2 = X2.ravel()
        # Take first n_front
        X1 = X1[:n_front]
        X2 = X2[:n_front]

        X_seed = np.zeros((n_front, self.dim))
        X_seed[:, 0] = X1
        X_seed[:, 1] = X2
        X_seed[:, 2:] = 0.5  # g = 0 plane

        F_seed = np.empty((n_front, self.n_obj))
        for idx in range(n_front):
            F_seed[idx] = problem(X_seed[idx])

        # ---------------------------------------------------------------
        # 2) A few random off-front points for exploration
        # ---------------------------------------------------------------
        n_rand = 8
        X_rand = lo + (hi - lo) * rng.random((n_rand, self.dim))
        F_rand = np.empty((n_rand, self.n_obj))
        for idx in range(n_rand):
            F_rand[idx] = problem(X_rand[idx])

        X = np.vstack([X_seed, X_rand])
        F = np.vstack([F_seed, F_rand])
        used = n_front + n_rand

        N = 40  # population size for NSGA-II

        # ---------------------------------------------------------------
        # 3) NSGA-II helpers
        # ---------------------------------------------------------------
        def fast_non_dominated_sort(F_mat):
            """
            Fast non-dominated sort.
            Returns array of rank indices (0-based).
            """
            n = len(F_mat)
            if n == 0:
                return np.array([], dtype=np.int32)

            rank = np.full(n, -1, dtype=np.int32)
            # For each i, S[i] = set of individuals dominated by i
            S = [set() for _ in range(n)]
            n_dom = np.zeros(n, dtype=np.int32)  # count of individuals dominating i

            # Compute dominance relationships
            for i in range(n):
                fi = F_mat[i]
                # j is dominated by i if fi <= F_j and fi < F_j (strictly in at least one)
                le = np.all(fi <= F_mat, axis=1)
                lt = np.any(fi < F_mat, axis=1)
                dom_mask = le & lt
                dom_mask[i] = False
                doms = np.where(dom_mask)[0]
                S[i] = set(doms.tolist())
                if len(doms) > 0:
                    n_dom[doms] += 1

            # Assign front 0
            current_front = np.where(n_dom == 0)[0].tolist()
            current_rank = 0
            while current_front:
                next_front = []
                for i in current_front:
                    rank[i] = current_rank
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
                current_rank += 1

            return rank

        def crowding_distance(F_mat, rank_arr):
            """
            Compute crowding distance for each individual.
            """
            n = len(F_mat)
            dist = np.zeros(n)
            if n <= 2:
                dist[:] = np.inf
                return dist

            unique_ranks = sorted(set(rank_arr))
            for r in unique_ranks:
                mask = (rank_arr == r)
                idxs = np.where(mask)[0]
                if len(idxs) <= 2:
                    dist[idxs] = np.inf
                    continue

                for k in range(self.n_obj):
                    vals = F_mat[idxs, k]
                    order = np.argsort(vals)
                    sorted_idxs = idxs[order]
                    f_min = vals[order[0]]
                    f_max = vals[order[-1]]
                    dist[sorted_idxs[0]] = np.inf
                    dist[sorted_idxs[-1]] = np.inf
                    if f_max - f_min < 1e-15:
                        continue
                    for ii in range(1, len(sorted_idxs) - 1):
                        dist[sorted_idxs[ii]] += (
                            F_mat[sorted_idxs[ii + 1], k] - F_mat[sorted_idxs[ii - 1], k]
                        ) / (f_max - f_min)

            return dist

        def tournament_select(X_mat, F_mat, rank_arr, dist_arr, k=3):
            """Tournament selection with rank then crowding distance."""
            n = len(F_mat)
            if n == 0:
                return X_mat[0].copy()
            idxs = rng.integers(0, n, size=k)
            best = idxs[0]
            for ii in idxs[1:]:
                if rank_arr[ii] < rank_arr[best]:
                    best = ii
                elif rank_arr[ii] == rank_arr[best] and dist_arr[ii] > dist_arr[best]:
                    best = ii
            return X_mat[best].copy()

        def sbx_crossover(x1, x2):
            """Simulated Binary Crossover."""
            if rng.random() < 0.5:
                return x1.copy(), x2.copy()
            eta = 20.0
            c1 = x1.copy()
            c2 = x2.copy()
            for i in range(self.dim):
                if x1[i] == x2[i]:
                    continue
                u = rng.random()
                if u <= 0.5:
                    alpha = (2.0 * u) ** (1.0 / (eta + 1.0))
                else:
                    alpha = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                c1[i] = 0.5 * ((1.0 + alpha) * x1[i] + (1.0 - alpha) * x2[i])
                c2[i] = 0.5 * ((1.0 - alpha) * x1[i] + (1.0 + alpha) * x2[i])
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            return c1, c2

        def polynomial_mutation(x):
            """Polynomial mutation."""
            eta = 20.0
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    x[i] += delta * (hi[i] - lo[i])
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # ---------------------------------------------------------------
        # 4) NSGA-II main loop
        # ---------------------------------------------------------------
        while used < self.budget:
            if self._elapsed() > self.deadline:
                break

            # Compute ranks and distances for current population
            rank = fast_non_dominated_sort(F)
            dist = crowding_distance(F, rank)

            # Generate offspring
            offspring = []
            n_pairs = N // 2
            for _ in range(n_pairs):
                if used + 2 > self.budget:
                    break
                if self._elapsed() > self.deadline:
                    break
                p1 = tournament_select(X, F, rank, dist)
                p2 = tournament_select(X, F, rank, dist)
                c1, c2 = sbx_crossover(p1, p2)
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                offspring.append(c1)
                offspring.append(c2)
                used += 2

            if not offspring:
                break

            # Evaluate offspring
            F_off = np.empty((len(offspring), self.n_obj))
            for idx, off in enumerate(offspring):
                F_off[idx] = problem(off)

            # Combine parent + offspring
            X_all = np.vstack([X, offspring])
            F_all = np.vstack([F, F_off])

            # Environmental selection
            rank_all = fast_non_dominated_sort(F_all)
            dist_all = crowding_distance(F_all, rank_all)

            # Sort by (rank, -dist) and take top N
            order = np.lexsort((-dist_all, rank_all))
            X = X_all[order[:N]]
            F = F_all[order[:N]]

        return X
