"""Improved multi-objective optimizer for WFG7."""
import numpy as np
import time


class Optimizer:
    """
    NSGA-II-based multi-objective optimizer with island model.
    Uses non-dominated sorting, crowding distance, and SBX crossover.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Island configuration
        n_islands = 3
        pop_size = 20  # 60 total initial, leaving 140 for evolution

        if n_islands * pop_size >= budget:
            n_islands = 2
            pop_size = max(10, budget // 6)

        total_init = n_islands * pop_size
        used = 0

        # Initialize populations
        populations = []  # list of (X, F) per island
        rng_main = np.random.RandomState(42)

        for i in range(n_islands):
            X = lo + range_ * rng_main.rand(pop_size, dim)
            F = np.zeros((pop_size, n_obj))
            for j in range(pop_size):
                F[j] = problem(X[j])
            used += pop_size
            populations.append((X, F))

        # NSGA-II parameters
        crossover_prob = 0.9
        mutation_prob = 1.0 / dim
        sbx_eta = 15.0
        mut_eta = 20.0
        migration_interval = 8

        gen_count = 0

        while used < budget:
            if time.time() > deadline:
                break

            for island_idx in range(n_islands):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                X, F = populations[island_idx]
                n = len(X)

                # Non-dominated sorting
                fronts = self._fast_non_dominated_sort(F)

                # Crowding distance
                cd = np.zeros(n)
                for front in fronts:
                    if len(front) <= 2:
                        cd[front] = np.inf
                        continue
                    for m in range(n_obj):
                        front_sorted = sorted(front, key=lambda idx: F[idx, m])
                        cd[front_sorted[0]] = np.inf
                        cd[front_sorted[-1]] = np.inf
                        fmin = F[front_sorted[0], m]
                        fmax = F[front_sorted[-1], m]
                        if fmax - fmin < 1e-14:
                            continue
                        for k in range(1, len(front_sorted) - 1):
                            idx = front_sorted[k]
                            cd[idx] += (F[front_sorted[k + 1], m] - F[front_sorted[k - 1], m]) / (fmax - fmin)

                # Selection: tournament of size 2
                def tournament_select():
                    a = rng_main.randint(n)
                    b = rng_main.randint(n)
                    # Compare by front rank then crowding distance
                    rank_a = self._get_rank(fronts, a)
                    rank_b = self._get_rank(fronts, b)
                    if rank_a < rank_b:
                        return a
                    elif rank_b < rank_a:
                        return b
                    else:
                        return a if cd[a] > cd[b] else b

                # Create offspring
                child_X = []
                child_F = []

                while len(child_X) < n and used < budget:
                    p1 = tournament_select()
                    p2 = tournament_select()

                    if rng_main.rand() < crossover_prob:
                        c1, c2 = self._sbx_crossover(X[p1], X[p2], lo, hi, range_, sbx_eta, rng_main)
                    else:
                        c1 = X[p1].copy()
                        c2 = X[p2].copy()

                    c1 = self._poly_mutation(c1, lo, hi, range_, mut_eta, mutation_prob, rng_main)
                    c2 = self._poly_mutation(c2, lo, hi, range_, mut_eta, mutation_prob, rng_main)

                    f1 = problem(c1)
                    used += 1
                    if time.time() > deadline:
                        child_X.append(c1)
                        child_F.append(f1)
                        break

                    f2 = problem(c2)
                    used += 1

                    child_X.append(c1)
                    child_F.append(f1)
                    child_X.append(c2)
                    child_F.append(f2)

                if not child_X:
                    continue

                # Combine parent and offspring
                combined_X = np.vstack([X] + [np.array(child_X)])
                combined_F = np.vstack([F] + [np.array(child_F)])
                n_combined = len(combined_X)

                # Non-dominated sorting on combined
                comb_fronts = self._fast_non_dominated_sort(combined_F)
                comb_cd = np.zeros(n_combined)
                for front in comb_fronts:
                    if len(front) <= 2:
                        comb_cd[front] = np.inf
                        continue
                    for m in range(n_obj):
                        front_sorted = sorted(front, key=lambda idx: combined_F[idx, m])
                        comb_cd[front_sorted[0]] = np.inf
                        comb_cd[front_sorted[-1]] = np.inf
                        fmin = combined_F[front_sorted[0], m]
                        fmax = combined_F[front_sorted[-1], m]
                        if fmax - fmin < 1e-14:
                            continue
                        for k in range(1, len(front_sorted) - 1):
                            idx = front_sorted[k]
                            comb_cd[idx] += (combined_F[front_sorted[k + 1], m] - combined_F[front_sorted[k - 1], m]) / (fmax - fmin)

                # Select best n individuals
                selected = []
                for front in comb_fronts:
                    if len(selected) + len(front) <= n:
                        selected.extend(front)
                    else:
                        front_sorted = sorted(front, key=lambda idx: comb_cd[idx], reverse=True)
                        selected.extend(front_sorted[:n - len(selected)])
                        break

                populations[island_idx] = (combined_X[selected], combined_F[selected])

            gen_count += 1
            if gen_count % migration_interval == 0:
                self._migrate(populations)

        # Return best found (return value is ignored by evaluator)
        all_F = np.vstack([F for _, F in populations])
        best_idx = np.argmin(np.sum(all_F, axis=1))
        return all_F[best_idx]

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        n_obj = F.shape[1]
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dom[i] += 1

        current_front = [i for i in range(n) if n_dom[i] == 0]
        k = 0
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
            k += 1

        return fronts

    def _get_rank(self, fronts, idx):
        for rank, front in enumerate(fronts):
            if idx in front:
                return rank
        return len(fronts)

    def _sbx_crossover(self, p1, p2, lo, hi, range_, eta, rng):
        c1 = p1.copy()
        c2 = p2.copy()

        for i in range(len(p1)):
            if rng.rand() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if 2 * rng.rand() <= (1 + 1 / (eta + 1)):
                        beta = (2 * rng.rand()) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * rng.rand())) ** (1 / (eta + 1))

                    c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])

                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])

        return c1, c2

    def _poly_mutation(self, x, lo, hi, range_, eta, prob, rng):
        for i in range(len(x)):
            if rng.rand() < prob:
                if abs(hi[i] - lo[i]) < 1e-14:
                    continue
                delta1 = (x[i] - lo[i]) / range_[i]
                delta2 = (hi[i] - x[i]) / range_[i]

                u = rng.rand()
                if u < 0.5:
                    xy = 1 - delta1
                    val = 2 * u + (1 - 2 * u) * (xy ** (eta + 1))
                    delta_q = val ** (1 / (eta + 1)) - 1
                else:
                    xy = 1 - delta2
                    val = 2 * (1 - u) + 2 * (u - 0.5) * (xy ** (eta + 1))
                    delta_q = 1 - val ** (1 / (eta + 1))

                x[i] = x[i] + delta_q * range_[i]
                x[i] = np.clip(x[i], lo[i], hi[i])

        return x

    def _migrate(self, populations):
        n_islands = len(populations)
        for i in range(n_islands):
            X, F = populations[i]
            # Find best individual (lowest sum)
            best_idx = np.argmin(np.sum(F, axis=1))

            # Migrate to next island
            target = (i + 1) % n_islands
            tX, tF = populations[target]
            # Replace worst in target
            worst_idx = np.argmax(np.sum(tF, axis=1))
            tX[worst_idx] = X[best_idx].copy()
            tF[worst_idx] = F[best_idx].copy()
            populations[target] = (tX, tF)
