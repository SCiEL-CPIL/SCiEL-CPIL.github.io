"""Multi-objective optimizer on WFG4: NSGA-II style with time deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective evolutionary optimizer with deadline guard."""

    def __init__(self, dim, n_obj, budget):
        self.dim = dim
        self.n_obj = n_obj
        self.budget = budget

    # ---- helpers -------------------------------------------------------
    def _non_dominated_sort(self, F):
        """Return list of fronts (lists of indices) and per-point rank."""
        n = len(F)
        S = [set() for _ in range(n)]   # S[i] = indices dominated by i
        n_count = np.zeros(n, dtype=int)
        rank = np.full(n, -1, dtype=int)
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # i dominates j ?
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_count[i] += 1
        # initial front
        f0 = [i for i in range(n) if n_count[i] == 0]
        for i in f0:
            rank[i] = 0
        fronts.append(f0)
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_count[j] -= 1
                    if n_count[j] == 0:
                        rank[j] = k + 1
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)
        return fronts, rank

    def _crowding(self, F, indices):
        """Crowding distance for the given indices within F."""
        m = len(indices)
        cd = np.zeros(m)
        if m <= 2:
            return cd
        for d in range(self.n_obj):
            vals = F[indices, d]
            mn, mx = vals.min(), vals.max()
            rng = mx - mn
            order = np.argsort(vals)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            if rng > 1e-15:
                for k in range(1, m - 1):
                    cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / rng
        return cd

    def _binary_tournament(self, rank, cd_map, rng):
        a = int(rng.integers(self.pop_size))
        b = int(rng.integers(self.pop_size))
        ra, rb = rank[a], rank[b]
        if ra < rb:
            return a
        if ra > rb:
            return b
        # same rank -> use crowding
        cda = cd_map.get(a, 0.0)
        cdb = cd_map.get(b, 0.0)
        return a if cda > cdb else b

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # deterministic RNG for reproducibility
        rng = np.random.default_rng(12345)

        # population size: balance search quality vs. per-iteration cost
        self.pop_size = min(40, max(10, budget // 5))

        # initial population
        X = lo + (hi - lo) * rng.random((self.pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = self.pop_size

        # track all evaluated (for hypervolume we only need the returned set,
        # but we keep the best non-dominated archive)
        all_F = F.copy()

        while used < budget:
            # deadline guard
            if time.time() - start_time > deadline:
                break

            fronts, rank = self._non_dominated_sort(F)

            # crowding distance per front
            cd_map = {}
            for front in fronts:
                cd = self._crowding(F, front)
                for idx, c in zip(front, cd):
                    cd_map[idx] = c

            # build offspring
            offspring = []
            while len(offspring) < self.pop_size:
                if time.time() - start_time > deadline:
                    break
                pa = self._binary_tournament(rank, cd_map, rng)
                pb = self._binary_tournament(rank, cd_map, rng)

                # crossover (SBX-like blend)
                if rng.random() < 0.9:
                    beta = rng.random(dim)
                    child = beta * X[pa] + (1.0 - beta) * X[pb]
                else:
                    child = X[pa].copy()

                # adaptive mutation: larger early, smaller late
                frac = used / max(1, budget)
                scale = 0.3 * (1.0 - 0.5 * frac)
                if rng.random() < 0.8:
                    child = child + scale * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                offspring.append((child, fc))

            if not offspring:
                break

            off_X = np.array([o[0] for o in offspring])
            off_F = np.array([o[1] for o in offspring], dtype=float)

            # combine parents + offspring, select pop_size via NSGA-II
            comb_X = np.vstack([X, off_X])
            comb_F = np.vstack([F, off_F])
            c_fronts, c_rank = self._non_dominated_sort(comb_F)

            # fill selection
            sel = []
            for front in c_fronts:
                if len(sel) + len(front) <= self.pop_size:
                    sel.extend(front)
                else:
                    need = self.pop_size - len(sel)
                    cd = self._crowding(comb_F, front)
                    order = np.argsort(-cd)
                    sel.extend(list(np.array(front)[order[:need]]))
                    break

            X = comb_X[sel]
            F = comb_F[sel]

            # update archive of best non-dominated seen so far
            all_F = np.vstack([all_F, off_F])

        # return the final non-dominated set (best result found)
        fronts, rank = self._non_dominated_sort(F)
        return X
# EVOLVE-BLOCK-END
