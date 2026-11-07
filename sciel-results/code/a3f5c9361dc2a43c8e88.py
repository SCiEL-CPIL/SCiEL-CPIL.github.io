"""Multi-objective optimizer for RE37 using Halton initialization, NSGA-II, and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer combining Halton space-filling design,
    NSGA-II evolutionary search, and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def _halton(self, n, dim):
        """Generate n Halton sequence points in [0,1)^dim."""
        points = np.zeros((n, dim))
        for i in range(1, n + 1):
            for d in range(dim):
                base = d + 2  # primes: 2, 3, 5, 7, ...
                f = 1.0
                result = 0.0
                j = i
                while j > 0:
                    f /= base
                    result += f * (j % base)
                    j //= base
                points[i - 1, d] = result
        return points

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 315.0
        self.eval_count = 0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        rng = np.random.default_rng(0)

        def eval_real(x):
            if self.eval_count >= self.budget:
                raise RuntimeError("budget exceeded")
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != dim:
                x = np.resize(x, dim)
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float).reshape(-1)
            self.eval_count += 1
            return f

        # Track all evaluated points
        self.X = []
        self.F = []

        def add_eval(x, f):
            x = np.clip(np.asarray(x, dtype=float).reshape(-1), lo, hi)
            f = np.asarray(f, dtype=float).reshape(-1)
            self.X.append(x.copy())
            self.F.append(f.copy())

        # --- Fast Non-dominated Sorting ---
        def fast_non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return [], np.array([], dtype=int)

            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.full(n, -1, dtype=int)

            for i in range(n):
                for j in range(i + 1, n):
                    fi = F[i]
                    fj = F[j]
                    if np.all(fi <= fj) and np.any(fi < fj):
                        S[i].append(j)
                        n_dom[j] += 1
                    if np.all(fj <= fi) and np.any(fj < fi):
                        S[j].append(i)
                        n_dom[i] += 1

            fronts = []
            current_front = [i for i in range(n) if n_dom[i] == 0]

            while current_front:
                fronts.append(current_front[:])
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front

            for front_idx, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = front_idx

            return fronts, rank

        def crowding_distance(front_indices, F):
            if len(front_indices) <= 2:
                return {i: float('inf') for i in front_indices}

            n = len(front_indices)
            dist = {i: 0.0 for i in front_indices}

            for k in range(n_obj):
                vals = [F[i][k] for i in front_indices]
                vmin = min(vals)
                vmax = max(vals)
                if vmax - vmin < 1e-15:
                    continue
                sorted_idx = sorted(front_indices, key=lambda i: F[i][k])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (F[sorted_idx[i + 1]][k] - F[sorted_idx[i - 1]][k]) / (vmax - vmin)

            return dist

        # --- Reference point for validity checking ---
        ref_point = np.array([1.1, 1.3, 1.1])

        # --- Phase 1: Halton sequence initialization ---
        n_init = min(40, self.budget // 5)
        halton_pts = self._halton(n_init, dim)
        X_init = lo + halton_pts * span

        # Add a few structured points for better coverage
        structured = [
            (lo + hi) / 2.0,
            lo + 0.25 * span,
            lo + 0.75 * span,
        ]
        for x in structured:
            X_init = np.vstack([X_init, x])

        # Evaluate initial set
        for x in X_init:
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            try:
                x = np.clip(x, lo, hi)
                f = eval_real(x)
                add_eval(x, f)
            except Exception:
                break

        if len(self.X) == 0:
            return None

        # --- Phase 2: NSGA-II Evolutionary Loop ---
        population_size = 25
        offspring_size = 35
        max_generations = 40

        for gen in range(max_generations):
            if time.time() > deadline:
                break
            if self.eval_count >= self.budget:
                break

            n_current = len(self.X)
            if n_current == 0:
                break

            # Take a subset as current population for selection
            pop_size = min(population_size, n_current)
            pop_indices = rng.choice(n_current, size=pop_size, replace=False)
            pop_X = np.array([self.X[i] for i in pop_indices])
            pop_F = np.array([self.F[i] for i in pop_indices])

            # Evaluate non-dominance ranks
            fronts, rank = fast_non_dominated_sort(pop_F)

            # Binary tournament selection
            def tournament_select(F, rank):
                i = rng.integers(0, len(F))
                j = rng.integers(0, len(F))
                if rank[i] < rank[j]:
                    return i
                elif rank[j] < rank[i]:
                    return j
                else:
                    return i if rng.random() < 0.5 else j

            # Generate offspring
            offspring_X = []
            offspring_F = []

            for _ in range(offspring_size):
                if time.time() > deadline:
                    break
                if self.eval_count >= self.budget:
                    break

                i = tournament_select(pop_F, rank)
                j = tournament_select(pop_F, rank)

                # Blend with Gaussian perturbation
                alpha = rng.random()
                child_x = alpha * pop_X[i] + (1 - alpha) * pop_X[j]

                # Add mutation
                mutation_rate = 0.3
                if rng.random() < mutation_rate:
                    child_x = child_x + rng.normal(0, 0.15, dim)

                child_x = np.clip(child_x, lo, hi)

                try:
                    f = eval_real(child_x)
                    add_eval(child_x, f)
                    offspring_X.append(child_x)
                    offspring_F.append(f)
                except Exception:
                    break

            if len(offspring_X) == 0:
                continue

            # Combine parent and offspring populations
            combined_F = np.array(self.F)

            # Get non-dominance ranks for combined population
            fronts, rank = fast_non_dominated_sort(combined_F)

            # Local search on Pareto front points
            if len(fronts) > 0:
                front0 = fronts[0]
                if len(front0) > 0:
                    n_local = min(3, len(front0))
                    local_idx = rng.choice(front0, size=n_local, replace=False)

                    for idx in local_idx:
                        if self.eval_count >= self.budget or time.time() > deadline:
                            break

                        x0 = self.X[idx].copy()
                        f0 = self.F[idx].copy()

                        # Simple coordinate descent
                        for step in range(6):
                            if self.eval_count >= self.budget or time.time() > deadline:
                                break
                            best_x = x0
                            best_f = f0

                            for d in range(dim):
                                for delta in [0.02, 0.05, 0.1, -0.02, -0.05, -0.1]:
                                    trial = best_x.copy()
                                    trial[d] += delta
                                    trial = np.clip(trial, lo, hi)

                                    try:
                                        f_trial = eval_real(trial)
                                        if np.all(f_trial <= best_f) and np.any(f_trial < best_f):
                                            best_x = trial
                                            best_f = f_trial
                                    except Exception:
                                        continue

                            if np.array_equal(best_x, x0):
                                break
                            x0 = best_x
                            f0 = best_f

                        add_eval(x0, f0)

        # --- Phase 3: Final local search on all Pareto front points ---
        if len(self.X) > 0:
            F_arr = np.array(self.F)
            fronts, rank = fast_non_dominated_sort(F_arr)

            if len(fronts) > 0:
                front0 = fronts[0]
                n_final = min(5, len(front0))
                if n_final > 0:
                    final_idx = rng.choice(front0, size=n_final, replace=False)
                    for idx in final_idx:
                        if self.eval_count >= self.budget or time.time() > deadline:
                            break

                        x0 = self.X[idx].copy()
                        f0 = self.F[idx].copy()

                        # Fine-tune with small steps
                        for step in range(4):
                            if self.eval_count >= self.budget or time.time() > deadline:
                                break
                            improved = False
                            for d in range(dim):
                                for delta in [0.01, 0.03, -0.01, -0.03]:
                                    trial = x0.copy()
                                    trial[d] += delta
                                    trial = np.clip(trial, lo, hi)
                                    try:
                                        f_trial = eval_real(trial)
                                        if np.all(f_trial <= f0) and np.any(f_trial < f0):
                                            x0 = trial
                                            f0 = f_trial
                                            improved = True
                                            break
                                    except Exception:
                                        continue
                                if improved:
                                    break
                            if not improved:
                                break

                        add_eval(x0, f0)

        return None
# EVOLVE-BLOCK-END
