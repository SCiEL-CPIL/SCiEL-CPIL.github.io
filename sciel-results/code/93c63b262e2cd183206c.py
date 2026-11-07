"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Memetic NSGA-II hybrid for WFG9 with adaptive mutation and local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # Initial population
        pop_size = min(30, self.budget // 3)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all non-dominated points for hypervolume
        all_points = list(F)

        # NSGA-II parameters
        n_gen = 0
        while used < self.budget and time.time() < deadline:
            n_gen += 1

            # Check time budget
            if time.time() > deadline - 10:
                break

            # Fast non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)

            # Crowding distance within front
            def crowding_distance(front_idx):
                f = F[front_idx]
                m = len(front_idx)
                if m <= 2:
                    return np.full(m, np.inf)
                cd = np.zeros(m)
                for k in range(self.n_obj):
                    sorted_idx = np.argsort(f[:, k])
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    fmin, fmax = f[sorted_idx[0], k], f[sorted_idx[-1], k]
                    if fmax - fmin < 1e-15:
                        continue
                    for i in range(1, m - 1):
                        cd[sorted_idx[i]] += (f[sorted_idx[i + 1], k] - f[sorted_idx[i - 1], k]) / (fmax - fmin)
                return cd

            # Build mating pool: select parents from top fronts
            parents_idx = []
            for front in fronts:
                if len(parents_idx) + len(front) >= pop_size:
                    cd = crowding_distance(front)
                    # Select from this front based on crowding
                    n_need = pop_size - len(parents_idx)
                    if n_need > 0:
                        # Sort by crowding distance descending
                        order = np.argsort(-cd)
                        parents_idx.extend(front[order[:n_need]])
                    break
                parents_idx.extend(front)

            # Generate offspring via crossover + mutation
            n_offspring = pop_size
            children_X = []
            for _ in range(n_offspring):
                # Tournament selection
                p1 = self._tournament_select(fronts, pop_size, rng)
                p2 = self._tournament_select(fronts, pop_size, rng)

                # Simulated binary crossover
                if rng.random() < 0.9:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Adaptive mutation
                c1 = self._adaptive_mutate(c1, lo, hi, rng, n_gen)
                c2 = self._adaptive_mutate(c2, lo, hi, rng, n_gen)

                children_X.append(c1)
                children_X.append(c2)
                used += 2
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break

            if not children_X:
                break

            # Evaluate children
            children_F = np.array([problem(x) for x in children_X])
            all_points.extend(children_F.tolist())

            # Combine parent and child populations
            combined_X = np.vstack([X, np.array(children_X)])
            combined_F = np.vstack([F, children_F])

            # Non-dominated sorting of combined
            combined_fronts = self._fast_non_dominated_sort(combined_F)

            # Select next generation
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X.append(combined_X[front])
                    new_F.append(combined_F[front])
                else:
                    n_need = pop_size - len(new_X)
                    cd = crowding_distance(front)
                    order = np.argsort(-cd)
                    selected = front[order[:n_need]]
                    new_X.append(combined_X[selected])
                    new_F.append(combined_F[selected])
                    break

            if new_X:
                X = np.vstack(new_X)
                F = np.vstack(new_F)

            # Local search on best few points (memetic step)
            if n_gen % 5 == 0 and time.time() < deadline - 20:
                X, F, used = self._local_search(X, F, problem, lo, hi, rng, used, deadline)

        return X

    def _fast_non_dominated_sort(self, F):
        """Return list of fronts, each a list of indices."""
        n = len(F)
        S = [[] for _ in range(n)]  # Dominated by
        n_dominates = np.zeros(n, dtype=int)
        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominates[i] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dominates[j] += 1
            if n_dominates[i] == 0:
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)

        return fronts[:-1] if fronts[-1] else fronts

    def _tournament_select(self, fronts, pop_size, rng):
        """Select one individual via tournament on front rank."""
        idx1 = rng.integers(len(fronts[0])) if fronts[0] else 0
        idx2 = rng.integers(len(fronts[0])) if fronts[0] else 0
        # Find which front each belongs to
        f1 = self._front_rank(fronts, idx1)
        f2 = self._front_rank(fronts, idx2)
        if f1 < f2:
            return fronts[0][idx1] if idx1 < len(fronts[0]) else fronts[0][0]
        elif f2 < f1:
            return fronts[0][idx2] if idx2 < len(fronts[0]) else fronts[0][0]
        else:
            return fronts[0][rng.integers(len(fronts[0]))]

    def _front_rank(self, fronts, idx):
        for i, front in enumerate(fronts):
            if idx in front:
                return i
        return len(fronts) - 1

    def _sbx_crossover(self, p1, p2, lo, hi, rng):
        """Simulated binary crossover."""
        c1, c2 = p1.copy(), p2.copy()
        for i in range(self.dim):
            if rng.random() < 0.5:
                if abs(p2[i] - p1[i]) > 1e-15:
                    beta = 1.0 / (1.0 + 2.0 * abs((p1[i] - p2[i]) / (hi[i] - lo[i] + 1e-15)))
                    if rng.random() <= 0.5:
                        alpha = (2.0 * beta ** (2.0 + 1.0 / (2.0 + 1.0)) - 1.0)
                    else:
                        alpha = (2.0 * (1.0 - beta) ** (2.0 + 1.0 / (2.0 + 1.0)) - 1.0)
                    u = rng.random()
                    if u <= 1.0 / (alpha + 1.0):
                        beta_q = (u * (alpha + 1.0)) ** (1.0 / (2.0 + 1.0 / (2.0 + 1.0)))
                    else:
                        beta_q = (1.0 / (2.0 - u * (alpha + 1.0))) ** (1.0 / (2.0 + 1.0 / (2.0 + 1.0)))
                    c1[i] = 0.5 * ((1.0 + beta_q) * p1[i] + (1.0 - beta_q) * p2[i])
                    c2[i] = 0.5 * ((1.0 - beta_q) * p1[i] + (1.0 + beta_q) * p2[i])
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _adaptive_mutate(self, x, lo, hi, rng, gen):
        """Polynomial mutation with adaptive rate."""
        rate = 1.0 / (self.dim + 1.0)
        eta = 20.0
        for i in range(self.dim):
            if rng.random() < rate:
                if hi[i] - lo[i] < 1e-15:
                    continue
                delta1 = (x[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - x[i]) / (hi[i] - lo[i])
                u = rng.random()
                if u < 0.5:
                    xy = (1.0 + 2.0 * delta1) ** (1.0 / (eta + 1.0)) if delta1 > 0 else 1.0
                    if 2.0 * u * xy - 1.0 > 0:
                        beta = (2.0 * u * xy - 1.0) ** (1.0 / (eta + 1.0))
                    else:
                        beta = 0.0
                else:
                    xy = (1.0 + 2.0 * delta2) ** (1.0 / (eta + 1.0)) if delta2 > 0 else 1.0
                    if 1.0 - 2.0 * (1.0 - u) * xy > 0:
                        beta = (1.0 - 2.0 * (1.0 - u) * xy) ** (1.0 / (eta + 1.0))
                    else:
                        beta = 0.0
                step = (hi[i] - lo[i]) * beta
                if rng.random() < 0.5:
                    x[i] -= step
                else:
                    x[i] += step
        return np.clip(x, lo, hi)

    def _local_search(self, X, F, problem, lo, hi, rng, used, deadline):
        """Perform local search on top individuals."""
        n_ls = min(5, len(X))
        # Pick the best n_ls based on sum of objectives
        order = np.argsort(np.sum(F, axis=1))
        best_idx = order[:n_ls]

        for idx in best_idx:
            if used >= self.budget or time.time() > deadline - 10:
                break

            x = X[idx].copy()
            fx = F[idx].copy()
            step_size = 0.05 * (hi - lo)

            for _ in range(20):
                if used >= self.budget or time.time() > deadline - 10:
                    break
                # Try random perturbation
                dx = step_size * rng.standard_normal(self.dim)
                x_new = np.clip(x + dx, lo, hi)
                if np.allclose(x_new, x):
                    continue
                fx_new = problem(x_new)
                used += 1
                # Accept if non-dominated or better in at least one objective
                if np.any(fx_new < fx):
                    # Check if it dominates
                    if np.all(fx_new <= fx):
                        x = x_new
                        fx = fx_new
                        step_size *= 0.9
                    elif np.any(fx_new < fx) and not np.all(fx <= fx_new):
                        # Partial improvement, accept with probability
                        if rng.random() < 0.3:
                            x = x_new
                            fx = fx_new
                else:
                    step_size *= 0.9

            X[idx] = x
            F[idx] = fx

        return X, F, used
# EVOLVE-BLOCK-END
