"""Multi-objective optimizer for RE37: DE-based with local polish and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution + Nelder-Mead polish for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size for DE
        pop_size = min(16, max(4, budget // 12))

        # Initialize population
        np.random.seed(0)  # Use the seed set by evaluator
        X = lo + (hi - lo) * np.random.random((pop_size, d))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # DE parameters
        F_factor = 0.8
        CR = 0.7

        # Track all evaluated points for hypervolume
        all_points = list(F)

        def is_dominated(a, b):
            """Check if a dominates b (a is better in all, strictly better in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(points):
            """Return indices of non-dominated points."""
            n = len(points)
            if n == 0:
                return []
            nd_idx = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if is_dominated(points[j], points[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_idx.append(i)
            return nd_idx

        def nelder_mead(x0, f0, bounds, max_iter=100):
            """Nelder-Mead local optimization on the sum of objectives."""
            d = len(x0)
            # Initialize simplex
            simplex = [x0.copy()]
            f_vals = [np.sum(f0)]
            for i in range(d):
                step = x0.copy()
                step[i] += 0.05 * (bounds[i][1] - bounds[i][0])
                step = np.clip(step, lo, hi)
                simplex.append(step)
                f_vals.append(np.sum(problem(step)))
            used_nm = d + 1

            alpha, gamma, rho = 1.0, 2.0, 0.5
            for _ in range(max_iter):
                if time.time() > deadline - 30:
                    break
                # Sort simplex
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                fr = np.sum(problem(xr))
                used_nm += 1

                if fr < f_vals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lo, hi)
                    fe = np.sum(problem(xe))
                    used_nm += 1
                    if fe < fr:
                        simplex[-1] = xe
                        f_vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_vals[-1] = fr
                elif fr < f_vals[-2]:
                    simplex[-1] = xr
                    f_vals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, lo, hi)
                    fc = np.sum(problem(xc))
                    used_nm += 1
                    if fc < f_vals[-1]:
                        simplex[-1] = xc
                        f_vals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            f_vals[i] = np.sum(problem(simplex[i]))
                            used_nm += 1

            return simplex[0], used_nm

        # DE loop
        generation = 0
        while used < budget:
            if time.time() > deadline - 20:
                break

            # Get non-dominated set
            nd_idx = non_dominated_sort(F)
            if not nd_idx:
                nd_idx = list(range(len(F)))

            # Select parent from non-dominated set
            parent_idx = nd_idx[np.random.randint(len(nd_idx))]
            p1 = X[parent_idx]
            p2 = X[np.random.randint(pop_size)]
            p3 = X[np.random.randint(pop_size)]

            # DE mutation
            v = p1 + F_factor * (p2 - p3)
            v = np.clip(v, lo, hi)

            # Crossover
            trial = p1.copy()
            for i in range(d):
                if np.random.random() < CR or i == np.random.randint(d):
                    trial[i] = v[i]
            trial = np.clip(trial, lo, hi)

            ft = problem(trial)
            used += 1
            all_points.append(ft)

            # Replace worst point in population
            # Use a simple criterion: replace the point with highest sum of objectives
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = trial
            F[worst_idx] = ft

            generation += 1

            # Periodic local polish on non-dominated points
            if generation % 20 == 0 and used < budget - 50:
                nd_idx = non_dominated_sort(F)
                # Polish up to 2 non-dominated points
                for idx in nd_idx[:2]:
                    if used >= budget - 10:
                        break
                    x_polished, nm_used = nelder_mead(X[idx], F[idx], bounds, max_iter=50)
                    used += nm_used
                    if used >= budget:
                        break
                    f_polished = problem(x_polished)
                    all_points.append(f_polished)
                    # Update population
                    X[idx] = x_polished
                    F[idx] = f_polished

        return X
# EVOLVE-BLOCK-END
