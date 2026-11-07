"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG9 with deadline guard and polish phase."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj

        # Budget allocation
        pop_size = min(30, self.budget // 3)
        if pop_size < 5:
            pop_size = 5

        # RNG
        rng = np.random.default_rng(42)

        # ---------- helpers ----------
        def dominates(a, b):
            """Check if a dominates b (all objectives <= and at least one <)."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            nd = [0] * n
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts.append([i])
            k = 0
            while k < len(fronts):
                current = fronts[k]
                next_front = []
                for i in current:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
            return fronts

        def crowding_distance(F_sub):
            """Compute crowding distance for a set of points."""
            m = len(F_sub)
            if m <= 2:
                return np.full(m, np.inf)
            dist = np.zeros(m)
            for k in range(n_obj):
                order = np.argsort(F_sub[:, k])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                fmin, fmax = F_sub[order[0], k], F_sub[order[-1], k]
                if fmax - fmin < 1e-14:
                    continue
                for i in range(1, m - 1):
                    dist[order[i]] += (F_sub[order[i + 1], k] - F_sub[order[i - 1], k]) / (fmax - fmin)
            return dist

        def tournament_select(pop_X, pop_F, pop_fronts, pop_cd, k=2):
            """Select k individuals via tournament on (front, crowding)."""
            candidates = rng.integers(len(pop_X), size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if pop_fronts[c] < pop_fronts[best]:
                    best = c
                elif pop_fronts[c] == pop_fronts[best] and pop_cd[c] > pop_cd[best]:
                    best = c
            return pop_X[best].copy()

        # ---------- initialization ----------
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Compute fronts and crowding distances for initial population
        fronts = fast_non_dominated_sort(F)
        front_of = np.zeros(pop_size, dtype=int)
        for fi, front in enumerate(fronts):
            for idx in front:
                front_of[idx] = fi

        # Crowding distance per front
        cd = np.zeros(pop_size)
        for fi, front in enumerate(fronts):
            if len(front) > 0:
                cd[front] = crowding_distance(F[front])

        # Archive of all non-dominated solutions found so far
        archive_F = F[fronts[0]].copy()
        archive_X = X[fronts[0]].copy()

        # ---------- DE evolution ----------
        F_scale = 0.7
        CR = 0.6

        while used < self.budget:
            if time.time() > deadline:
                break

            # Pick parents via tournament
            p1 = tournament_select(X, F, front_of, cd)
            p2 = tournament_select(X, F, front_of, cd)
            p3 = tournament_select(X, F, front_of, cd)

            # DE mutation
            v = p1 + F_scale * (p2 - p3)
            # Binary crossover
            mask = rng.random(dim) < CR
            mask[rng.integers(dim)] = True  # ensure at least one gene
            child = np.where(mask, v, p1)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Update archive
            dominated = False
            for af in archive_F:
                if dominates(af, fc):
                    dominated = True
                    break

            if not dominated:
                # Add to archive and remove dominated
                new_archive_F = np.vstack([archive_F, [fc]])
                new_archive_X = np.vstack([archive_X, [child]])
                keep = []
                for j, af in enumerate(new_archive_F):
                    if not dominates(fc, af):
                        keep.append(j)
                if keep:
                    archive_F = new_archive_F[keep]
                    archive_X = new_archive_X[keep]
                else:
                    archive_F = np.array([fc])
                    archive_X = np.array([child])

            # Replace worst in population if child is better
            sums = np.sum(F, axis=1)
            worst = np.argmax(sums)
            if np.sum(fc) < sums[worst]:
                X[worst] = child
                F[worst] = fc
                # Recompute fronts and crowding
                fronts = fast_non_dominated_sort(F)
                front_of = np.zeros(pop_size, dtype=int)
                for fi, front in enumerate(fronts):
                    for idx in front:
                        front_of[idx] = fi
                cd = np.zeros(pop_size)
                for fi, front in enumerate(fronts):
                    if len(front) > 0:
                        cd[front] = crowding_distance(F[front])

        # ---------- Polish phase: local search from diverse archive points ----------
        if used < self.budget and len(archive_F) > 0:
            # Get diverse subset of the archive
            fronts = fast_non_dominated_sort(archive_F)
            first_front_idx = fronts[0]
            first_front_F = archive_F[first_front_idx]
            first_front_X = archive_X[first_front_idx]
            cd = crowding_distance(first_front_F)

            # Pick up to 8 diverse points
            order = np.argsort(-cd)
            num_polish = min(8, len(order))
            polish_indices = order[:num_polish]

            local_sigma = 0.03
            for pi in polish_indices:
                if used >= self.budget or time.time() > deadline:
                    break

                base_X = first_front_X[pi].copy()

                for step in range(4):
                    if used >= self.budget or time.time() > deadline:
                        break

                    candidate = base_X + local_sigma * (hi - lo) * rng.standard_normal(dim)
                    candidate = np.clip(candidate, lo, hi)
                    fc = problem(candidate)
                    used += 1

                    dominated = False
                    for af in archive_F:
                        if dominates(af, fc):
                            dominated = True
                            break

                    if not dominated:
                        new_archive_F = np.vstack([archive_F, [fc]])
                        new_archive_X = np.vstack([archive_X, [candidate]])
                        keep = []
                        for j, af in enumerate(new_archive_F):
                            if not dominates(fc, af):
                                keep.append(j)
                        if keep:
                            archive_F = new_archive_F[keep]
                            archive_X = new_archive_X[keep]
                        else:
                            archive_F = np.array([fc])
                            archive_X = np.array([candidate])

        return X
# EVOLVE-BLOCK-END
