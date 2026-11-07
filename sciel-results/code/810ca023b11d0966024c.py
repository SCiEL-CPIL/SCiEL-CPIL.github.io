"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer combining weighted-sum decomposition with
    evolutionary refinement and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # Store all evaluated points and their objective vectors
        all_X = []
        all_F = []
        eval_count = [0]

        def count_eval(x):
            if eval_count[0] >= self.budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=np.float64)
            all_X.append(x.copy())
            all_F.append(f.copy())
            eval_count[0] += 1
            return f

        # ================================================================
        # Phase 1: Initial sampling - random + Latin hypercube for coverage
        # ================================================================
        n_init = min(15, self.budget // 5)

        # Latin hypercube for uniform coverage
        for i in range(self.dim):
            perm = rng.permutation(n_init)
            for j in range(n_init):
                x = lo + (hi - lo) * (perm[j] + rng.random()) / n_init
                count_eval(x)

        # Additional random points for diversity
        n_random_extra = max(5, self.budget // 10)
        for _ in range(n_random_extra):
            if time.time() > deadline or eval_count[0] >= self.budget:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            count_eval(x)

        # Helper: find non-dominated points
        def get_non_dominated():
            if len(all_F) == 0:
                return []
            F_arr = np.array(all_F)
            n = len(F_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    # Check if j dominates i
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            return [i for i in range(n) if not dominated[i]]

        # ================================================================
        # Phase 2: Weighted-sum L-BFGS-B decomposition
        # ================================================================
        # Generate diverse weight vectors on the simplex
        n_weights = 15
        weights = []

        # Extreme points (each objective individually)
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights.append(w)

        # Uniform random weights on simplex
        for _ in range(n_weights - self.n_obj):
            w = rng.random(self.n_obj)
            w /= w.sum()
            weights.append(w)

        weights = np.array(weights)

        for w in weights:
            if time.time() > deadline or eval_count[0] >= self.budget:
                break

            # Choose best starting point based on current weighted sum
            if len(all_F) > 0:
                F_arr = np.array(all_F)
                weighted_sums = F_arr @ w
                best_idx = np.argmin(weighted_sums)
                x0 = all_X[best_idx].copy()
            else:
                x0 = (lo + hi) / 2.0

            def obj(x, w=w):
                if eval_count[0] >= self.budget:
                    return np.inf
                fx = count_eval(x)
                if fx is None:
                    return np.inf
                return np.sum(fx * w)

            try:
                result = minimize(
                    obj, x0, method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 40, 'maxls': 20, 'maxfun': 30}
                )
                if result.success or result.fun < np.inf:
                    x_res = np.clip(result.x, lo, hi)
                    f_res = count_eval(x_res)
            except Exception:
                pass

        # ================================================================
        # Phase 3: Evolutionary refinement with adaptive mutation
        # ================================================================
        if eval_count[0] < self.budget and time.time() < deadline:
            # Build population from non-dominated points
            nd_idx = get_non_dominated()
            if len(nd_idx) == 0:
                nd_idx = list(range(len(all_F)))

            # Main evolutionary loop
            pop_size = max(8, min(20, (self.budget - eval_count[0]) // 6))
            iteration = 0
            max_iterations = 50

            while eval_count[0] < self.budget and time.time() < deadline and iteration < max_iterations:
                iteration += 1

                # Re-compute non-dominated set
                nd_idx = get_non_dominated()
                if len(nd_idx) < 2:
                    nd_idx = list(range(len(all_F)))

                # Adaptive mutation scale: starts large, decays over time
                progress = iteration / max_iterations
                mutation_scale = 0.3 * (1 - 0.7 * progress) * (hi - lo)

                # Pick two parents from non-dominated set
                p1 = nd_idx[rng.integers(len(nd_idx))]
                p2 = nd_idx[rng.integers(len(nd_idx))]

                # Crossover: blend parents
                x1 = all_X[p1]
                x2 = all_X[p2]
                alpha = rng.random(self.dim)
                child = alpha * x1 + (1 - alpha) * x2

                # Mutation
                child += mutation_scale * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

                fc = count_eval(child)
                if fc is None:
                    break

                # Occasionally try a local refinement around a good point
                if rng.random() < 0.3 and eval_count[0] < self.budget and time.time() < deadline:
                    # Pick a random non-dominated point for local search
                    if len(nd_idx) > 0:
                        local_idx = nd_idx[rng.integers(len(nd_idx))]
                        x0 = all_X[local_idx].copy()

                        # Try optimizing each objective separately
                        for obj_i in range(self.n_obj):
                            if time.time() > deadline or eval_count[0] >= self.budget:
                                break

                            def local_obj(x, obj_i=obj_i):
                                if eval_count[0] >= self.budget:
                                    return np.inf
                                fx = count_eval(x)
                                if fx is None:
                                    return np.inf
                                return fx[obj_i]

                            try:
                                minimize(
                                    local_obj, x0, method='L-BFGS-B',
                                    bounds=list(zip(lo, hi)),
                                    options={'maxiter': 15, 'maxfun': 20}
                                )
                            except Exception:
                                pass

        return all_X
# EVOLVE-BLOCK-END
