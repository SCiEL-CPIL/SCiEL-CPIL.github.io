"""Multi-objective optimizer for RE21 with hypervolume-guided DE + Powell refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hypervolume-driven multi-objective optimizer for RE21 (2 objectives, 4 vars)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.ref_point = np.array([3000.0, 0.06])

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = self.ref_point

        # Track all evaluated points
        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            all_X.append(x.copy())
            all_F.append(f)
            used += 1
            return f

        def non_dominated_mask(F_list):
            """Return boolean mask of non-dominated points."""
            if len(F_list) == 0:
                return np.array([], dtype=bool)
            F_arr = np.array(F_list, dtype=float)
            n = len(F_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            return ~dominated

        def compute_hv_2d(F_list):
            """Compute 2D hypervolume of F_list w.r.t. ref (only points strictly better)."""
            if len(F_list) == 0:
                return 0.0
            pts = np.array(F_list, dtype=float)
            mask = np.all(pts < ref, axis=1)
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            hv = 0.0
            for i in range(len(pts)):
                if i + 1 < len(pts):
                    w0 = pts[i + 1, 0] - pts[i, 0]
                else:
                    w0 = ref[0] - pts[i, 0]
                h1 = ref[1] - pts[i, 1]
                if h1 > 0 and w0 > 0:
                    hv += w0 * h1
            return hv

        # Phase 1: Stratified + random initialization
        n_init = 40
        rng = np.random.RandomState(42)

        # Stratified sampling
        n_strat = 30
        for i in range(n_strat):
            if used >= self.budget or time.time() > deadline:
                break
            x = np.zeros(self.dim)
            for d in range(self.dim):
                segment = int(i * self.dim / n_strat) % self.dim
                if d == segment:
                    frac = (i % (n_strat // self.dim + 1)) / (n_strat // self.dim + 1)
                    x[d] = lo[d] + frac * (hi[d] - lo[d])
                else:
                    x[d] = (lo[d] + hi[d]) / 2.0
            x = np.clip(x, lo, hi)
            evaluate(x)

        # Random points
        n_rand = 10
        for i in range(n_rand):
            if used >= self.budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(self.dim)
            evaluate(x)

        # Phase 2: Hypervolume-guided Adaptive DE
        pop_size = 20
        if used + pop_size > self.budget:
            pop_size = max(4, self.budget - used)
            if pop_size < 4:
                # Not enough budget for DE
                while used < self.budget and time.time() < deadline:
                    x = lo + (hi - lo) * np.random.random(self.dim)
                    evaluate(x)
                return all_X

        # Initialize population from non-dominated points + random
        mask_nd = non_dominated_mask(all_F)
        nd_indices = np.where(mask_nd)[0]
        pop = []

        for idx in nd_indices[:pop_size // 2]:
            pop.append(all_X[idx].copy())

        while len(pop) < pop_size and used < self.budget and time.time() < deadline:
            x = lo + (hi - lo) * np.random.random(self.dim)
            pop.append(x)
            evaluate(x)

        if len(pop) == 0:
            while used < self.budget and time.time() < deadline:
                x = lo + (hi - lo) * np.random.random(self.dim)
                evaluate(x)
            return all_X

        # DE loop with shrinking mutation factor
        max_generations = max(5, (self.budget - used) // (pop_size + 1))

        for gen in range(max_generations):
            if used >= self.budget or time.time() > deadline:
                break

            F_mut = 0.8 * (1.0 - gen / max_generations) + 0.2
            CR = 0.5

            # Evaluate current population's objectives (cache check)
            pop_F = []
            for x in pop:
                found = False
                for i, xi in enumerate(all_X):
                    if np.allclose(xi, x, atol=1e-10):
                        pop_F.append(all_F[i])
                        found = True
                        break
                if not found:
                    f = evaluate(x)
                    if f is None:
                        break
                    pop_F.append(f)

            if len(pop_F) < len(pop):
                break

            # Find non-dominated in population
            pop_mask = non_dominated_mask(pop_F)
            pop_front = np.where(pop_mask)[0]

            for i in range(len(pop)):
                if used >= self.budget or time.time() > deadline:
                    break

                # Differential evolution mutation
                if len(pop) >= 3:
                    candidates = [j for j in range(len(pop)) if j != i]
                    a, b, c = np.random.choice(candidates, size=min(3, len(candidates)), replace=False)
                    mutant = pop[a] + F_mut * (pop[b] - pop[c])
                else:
                    mutant = pop[i] + (hi - lo) * 0.1 * np.random.randn(self.dim)

                # Crossover
                trial = pop[i].copy()
                for d in range(self.dim):
                    if np.random.random() < CR or d == np.random.randint(self.dim):
                        trial[d] = mutant[d]

                trial = np.clip(trial, lo, hi)

                # Evaluate trial
                f_trial = evaluate(trial)
                if f_trial is None:
                    break

                # Selection: keep if it improves hypervolume or is non-dominated
                # Check if trial is non-dominated w.r.t. current global front
                global_mask = non_dominated_mask(all_F)
                global_front = np.where(global_mask)[0]

                is_dominated = False
                for idx in global_front:
                    if np.all(np.array(all_F[idx]) <= f_trial) and np.any(np.array(all_F[idx]) < f_trial):
                        is_dominated = True
                        break

                if not is_dominated:
                    pop[i] = trial

        # Phase 3: Powell local search on top non-dominated points
        remaining = self.budget - used
        if remaining > 10 and time.time() < deadline:
            mask_final = non_dominated_mask(all_F)
            nd_idx = np.where(mask_final)[0]

            # Sort by hypervolume contribution potential
            # Prioritize points with good balance
            def hv_contrib(idx):
                f = np.array(all_F[idx])
                if f[0] >= ref[0] or f[1] >= ref[1]:
                    return -1e10
                return (ref[0] - f[0]) * (ref[1] - f[1])

            nd_idx_sorted = sorted(nd_idx, key=hv_contrib, reverse=True)
            n_local = min(len(nd_idx_sorted), 5)

            for idx in nd_idx_sorted[:n_local]:
                if used >= self.budget or time.time() > deadline:
                    break

                x0 = all_X[idx].copy()
                bounds_opt = list(zip(lo, hi))

                def hv_surrogate(x):
                    nonlocal used
                    if used >= self.budget:
                        return 1e10
                    x_clipped = np.clip(x, lo, hi)
                    f = np.array(problem(x_clipped), dtype=float)
                    used += 1
                    if f[0] >= ref[0] or f[1] >= ref[1]:
                        return 1e10
                    return -((ref[0] - f[0]) * (ref[1] - f[1]))

                try:
                    res = minimize(
                        hv_surrogate,
                        x0,
                        method='Powell',
                        bounds=bounds_opt,
                        options={'maxfev': min(30, self.budget - used), 'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-6}
                    )
                    if res.fun < 1e9:
                        x_new = np.clip(res.x, lo, hi)
                        f_new = evaluate(x_new)
                except Exception:
                    continue

        # Return all evaluated points (evaluator records all)
        return all_X

# EVOLVE-BLOCK-END
