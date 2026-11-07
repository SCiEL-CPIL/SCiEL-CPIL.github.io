# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective NSGA-II optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin under 330s

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        N = 40

        # Evaluation counter
        eval_count = [0]

        def eval_point(x):
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            eval_count[0] += 1
            return f

        # ---- Phase 0: LHS initialization ----
        def lhs_sample(n, d, lo, hi):
            """Latin Hypercube Sampling."""
            X = np.zeros((n, d))
            for j in range(d):
                perm = np.random.permutation(n)
                X[:, j] = (perm + np.random.random(n)) / n
            return lo + X * (hi - lo)

        # Use numpy random (seeded by evaluator)
        X = lhs_sample(N, dim, lo, hi)
        F = np.array([eval_point(x) for x in X])
        used = N

        # ---- NSGA-II helpers ----
        def fast_non_dominated_sort(F):
            """Return list of fronts (each front is a list of indices)."""
            n = F.shape[0]
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_d = np.zeros(n, dtype=int)  # number of points dominating i
            fronts = []
            rank = np.zeros(n, dtype=int)

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_d[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_d[i] += 1

            # First front
            front0 = [i for i in range(n) if n_d[i] == 0]
            for i in front0:
                rank[i] = 0
            fronts.append(front0)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                if next_front:
                    k += 1
                    fronts.append(next_front)

            return fronts, rank

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            F_sub = F[indices]

            for m in range(n_obj):
                order = np.argsort(F_sub[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                f_range = F_sub[order[-1], m] - F_sub[order[0], m]
                if f_range < 1e-15:
                    continue
                for i in range(1, n - 1):
                    cd[order[i]] += (F_sub[order[i + 1], m] - F_sub[order[i - 1], m]) / f_range

            return cd

        def tournament_select(pop_F, pop_X, rank, cd, k=2):
            """Tournament selection, returns index of winner."""
            n = len(pop_X)
            a = np.random.randint(n)
            b = np.random.randint(n)
            if rank[a] < rank[b]:
                return a
            elif rank[b] < rank[a]:
                return b
            else:
                return a if cd[a] > cd[b] else b

        def crossover(x1, x2):
            """SBX crossover."""
            if np.random.random() < 0.9:
                eta = 20.0
                x = np.zeros_like(x1)
                for i in range(dim):
                    if np.random.random() < 0.5:
                        x[i] = x1[i]
                    else:
                        x[i] = x2[i]
                # Simple blend crossover
                alpha = np.random.random()
                x = alpha * x1 + (1 - alpha) * x2
                return np.clip(x, lo, hi)
            else:
                return x1.copy()

        def mutate(x, sigma):
            """Polynomial mutation."""
            eta = 20.0
            x_new = x.copy()
            for i in range(dim):
                if np.random.random() < 0.9:
                    u = np.random.random()
                    if u < 0.5:
                        val = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        val = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    x_new[i] += val * (hi[i] - lo[i])
            return np.clip(x_new, lo, hi)

        # ---- Main NSGA-II loop ----
        # Adaptive mutation rate
        base_sigma = 0.1
        sigma = base_sigma
        no_improve = 0
        best_hv = -np.inf

        def compute_hv(F, ref):
            """Simple hypervolume for 3D (for tracking)."""
            # Filter points dominated by reference
            mask = np.all(F < ref, axis=1)
            F_sub = F[mask]
            if len(F_sub) == 0:
                return 0.0
            # For 3D, use a simple grid-based approximation
            # This is just for tracking, not for the actual score
            return len(F_sub)

        while used < budget:
            if time.time() > deadline:
                break

            # Current population
            pop_X = X
            pop_F = F

            # Non-dominated sort
            fronts, rank = fast_non_dominated_sort(pop_F)

            # Crowding distance
            cd = np.zeros(len(pop_X))
            for front in fronts:
                if len(front) > 0:
                    cd_sub = crowding_distance(pop_F, front)
                    for idx, i in enumerate(front):
                        cd[i] = cd_sub[idx]

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = N

            while len(offspring_X) < n_offspring and used + n_offspring - len(offspring_X) <= budget:
                if time.time() > deadline:
                    break
                p1 = tournament_select(pop_F, pop_X, rank, cd)
                p2 = tournament_select(pop_F, pop_X, rank, cd)
                child = crossover(pop_X[p1], pop_X[p2])
                child = mutate(child, sigma)
                f = eval_point(child)
                offspring_X.append(child)
                offspring_F.append(f)
                used += 1

            if len(offspring_X) == 0:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            R_X = np.vstack([pop_X, offspring_X])
            R_F = np.vstack([pop_F, offspring_F])

            # Non-dominated sort on combined
            fronts_R, rank_R = fast_non_dominated_sort(R_F)

            # Select next generation
            next_X = []
            next_F = []
            for front in fronts_R:
                if len(next_X) + len(front) <= N:
                    for i in front:
                        next_X.append(R_X[i])
                        next_F.append(R_F[i])
                else:
                    # Need to truncate this front using crowding distance
                    remaining = N - len(next_X)
                    # Compute crowding distance for this front
                    cd_front = crowding_distance(R_F, front)
                    order = np.argsort(-cd_front)
                    for idx in order[:remaining]:
                        i = front[idx]
                        next_X.append(R_X[i])
                        next_F.append(R_F[i])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

            # Adapt sigma based on improvement
            # Track best scalarized objective
            fmin = F.min(axis=0)
            fmax = F.max(axis=0)
            fspan = (fmax - fmin) + 1e-12
            norm = (F - fmin) / fspan
            scalar = norm.sum(axis=1)
            current_best = scalar.min()

            if current_best < best_hv - 1e-8:
                best_hv = current_best
                no_improve = 0
                sigma = max(0.02, sigma * 0.95)
            else:
                no_improve += 1
                if no_improve > 10:
                    sigma = min(0.3, sigma * 1.1)
                    no_improve = 0

        return X
# EVOLVE-BLOCK-END
