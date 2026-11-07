import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT6 with deadline guard."""

    def __init__(self, dim, n_obj, budget):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin vs 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Deterministic RNG (seeds set externally via np.random.seed, but we use our own)
        rng = np.random.default_rng(0)

        pop_size = 24
        if pop_size > self.budget:
            pop_size = max(2, self.budget)

        # ---- Informed initialization ----
        # ZDT6: f1 = 1 - exp(-4*x1)*sin(10*pi*x1); f2 = 1 - (f1)^2
        # Pareto front: x1 in [0,1], x2..x10 arbitrary.
        # Sample x1 values that spread f1 across [0,1].
        def zdt6_f1(x1):
            return 1.0 - np.exp(-4.0 * x1) * np.sin(10.0 * np.pi * x1)

        # Build population: half informed (spread x1), half random
        half = pop_size // 2
        X = np.empty((pop_size, dim))
        # Informed: choose x1 to spread f1
        x1_samples = np.linspace(0.0, 1.0, half)
        for i in range(half):
            X[i, 0] = x1_samples[i]
            X[i, 1:] = rng.random(dim - 1)
        # Random fill
        X[half:] = lo + (hi - lo) * rng.random((pop_size - half, dim))

        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Keep an archive of all evaluated points for hypervolume (dedup by dominance handled by scoring)
        archive_X = X.copy()
        archive_F = F.copy()

        def dominated_mask(F):
            """Return boolean mask: True if point i is dominated by some other point."""
            n = len(F)
            dom = np.zeros(n, dtype=bool)
            for i in range(n):
                if dom[i]:
                    continue
                # i dominates j if F[i] <= F[j] all and < some
                better = (F <= F[i]).all(axis=1) & (F < F[i]).any(axis=1)
                better[i] = False
                if better.any():
                    dom[better] = True
            return dom

        def crowding_distance(F):
            """Crowding distance for a set of objective vectors."""
            n = len(F)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for k in range(n_obj):
                vals = F[:, k]
                mn, mx = vals.min(), vals.max()
                rng_k = mx - mn
                if rng_k < 1e-15:
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for j in range(1, n - 1):
                    cd[order[j]] += (vals[order[j + 1]] - vals[order[j - 1]]) / rng_k
            return cd

        def non_dominated_sort(F):
            """Return list of fronts (each a list of indices), front 0 = non-dominated."""
            n = len(F)
            fronts = []
            remaining = list(range(n))
            while remaining:
                rem_arr = np.array(remaining)
                dom = dominated_mask(F[rem_arr])
                nd_idx = rem_arr[~dom]
                fronts.append(list(nd_idx))
                remaining = [i for i in remaining if i not in set(nd_idx.tolist())]
                if len(remaining) == 0:
                    break
            return fronts

        def tournament_select(parents_F, parents_X, k=2):
            """Pick a parent via tournament on non-dominated + crowding."""
            idx = rng.integers(0, len(parents_F), size=k)
            cd = crowding_distance(parents_F[idx])
            best = idx[np.argmax(cd)]
            return parents_F[best], parents_X[best]

        # Adaptive mutation scale
        def mut_scale(used_frac):
            # Start wide, shrink for polish
            return 0.15 * (1.0 - used_frac) + 0.005

        while used < self.budget:
            if time.time() > deadline - 5.0:
                break

            # Selection: non-dominated sort current pop
            fronts = non_dominated_sort(F)
            # Parents = first front (non-dominated)
            parents_idx = fronts[0]
            parents_F = F[parents_idx]
            parents_X = X[parents_idx]

            # Create one child per generation step (to keep evaluation count aligned)
            # We'll create children and replace worst points.
            n_child = min(pop_size, self.budget - used)
            if n_child <= 0:
                break

            used_frac = used / float(self.budget)
            mscale = mut_scale(used_frac)

            # Crossover: blend two parents
            p1F, p1X = tournament_select(parents_F, parents_X)
            p2F, p2X = tournament_select(parents_F, parents_X)
            alpha = rng.random()
            child = alpha * p1X + (1.0 - alpha) * p2X
            # Mutation
            child = child + mscale * span * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = np.array(problem(child), dtype=float)
            used += 1

            # Update archive
            archive_X = np.vstack([archive_X, child[None, :]])
            archive_F = np.vstack([archive_F, fc[None, :]])

            # Replace worst point in population by sum-of-objectives (simple)
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc

        # ---- Final polish pass (if time remains) ----
        if used < self.budget and time.time() < deadline - 2.0:
            # Take the non-dominated set and do fine local search
            fronts = non_dominated_sort(F)
            nd_idx = fronts[0]
            polish = 3
            for pi in nd_idx:
                if used >= self.budget:
                    break
                if time.time() > deadline - 2.0:
                    break
                base = X[pi].copy()
                for _ in range(polish):
                    if used >= self.budget:
                        break
                    if time.time() > deadline - 2.0:
                        break
                    step = 0.01 * span * rng.standard_normal(dim)
                    cand = np.clip(base + step, lo, hi)
                    fcand = np.array(problem(cand), dtype=float)
                    used += 1
                    # Replace if it improves (dominates or better sum)
                    if np.all(fcand <= F[pi]) and np.any(fcand < F[pi]):
                        X[pi] = cand
                        F[pi] = fcand
                        archive_X = np.vstack([archive_X, cand[None, :]])
                        archive_F = np.vstack([archive_F, fcand[None, :]])
                    else:
                        # still record
                        archive_X = np.vstack([archive_X, cand[None, :]])
                        archive_F = np.vstack([archive_F, fcand[None, :]])
                        if np.sum(fcand) < np.sum(F[pi]):
                            X[pi] = cand
                            F[pi] = fcand

        return X
