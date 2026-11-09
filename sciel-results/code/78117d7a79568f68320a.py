"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer: Sobol init + adaptive scalarization + local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        used = 0
        all_X = []
        all_F = []

        def eval_point(x):
            nonlocal used
            x = np.asarray(x, dtype=float)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f.copy())
            return f

        def time_ok(margin=5.0):
            return (time.time() + margin) < deadline

        def get_nondominated(X, F):
            """Return indices of non-dominated points."""
            n = len(F)
            if n == 0:
                return []
            F_arr = np.array(F)
            nd_mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd_mask[i]:
                    continue
                for j in range(n):
                    if i == j or not nd_mask[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nd_mask[i] = False
                        break
            return np.where(nd_mask)[0].tolist()

        # Phase 1: Sobol quasi-random initialization for uniform coverage
        n_init = 50
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            X_init = sampler.random(n_init)
        except Exception:
            X_init = np.random.random((n_init, self.dim))

        X_init = lo + X_init * span

        for x in X_init:
            if used >= self.budget or not time_ok(10.0):
                break
            eval_point(x)

        # Get initial non-dominated set
        nd_idx = get_nondominated(all_X, all_F)
        nd_X = [all_X[i] for i in nd_idx]
        nd_F = [all_F[i] for i in nd_idx]

        # Phase 2: Systematic weight-vector scalarization with efficient local search
        # Generate well-distributed weight vectors on the simplex for 3 objectives
        # Use a structured grid: for 3 objectives, weights (a,b,c) with a+b+c=1
        weight_vectors = []
        # Include extreme and interior points
        n_per_axis = 5
        for i in range(n_per_axis + 1):
            for j in range(n_per_axis + 1 - i):
                a = i / n_per_axis
                b = j / n_per_axis
                c = 1.0 - a - b
                if a >= 0 and b >= 0 and c >= 0:
                    weight_vectors.append([a, b, c])
        # Add some random ones for diversity
        for _ in range(10):
            w = np.random.dirichlet(np.ones(self.n_obj))
            weight_vectors.append(w.tolist())

        weight_vectors = np.array(weight_vectors)
        n_weights = len(weight_vectors)

        # For each weight vector, do a short local optimization from best starting point
        evals_per_local = 8  # Keep local search very short to conserve budget
        max_local_calls = min(n_weights, (self.budget - used) // evals_per_local)

        for wi in range(max_local_calls):
            if used >= self.budget - 5 or not time_ok(10.0):
                break
            w = weight_vectors[wi]

            # Pick starting point from non-dominated set or random
            if nd_X and np.random.random() < 0.7:
                x0 = nd_X[np.random.randint(len(nd_X))].copy()
            else:
                x0 = lo + span * np.random.random(self.dim)

            def scalarized(x, w=w):
                f = eval_point(x)
                return float(np.dot(w, f))

            try:
                res = minimize(
                    scalarized,
                    x0,
                    method='Nelder-Mead',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 6, 'xatol': 1e-4, 'fatol': 1e-5, 'maxfev': evals_per_local}
                )
                # Evaluate the result
                if used < self.budget and time_ok(5.0):
                    x_res = np.clip(res.x, lo, hi)
                    eval_point(x_res)
            except Exception:
                pass

        # Update non-dominated set
        nd_idx = get_nondominated(all_X, all_F)
        nd_X = [all_X[i] for i in nd_idx]
        nd_F = [all_F[i] for i in nd_idx]

        # Phase 3: Gaussian perturbations around non-dominated points
        remaining = self.budget - used
        n_perturb = min(remaining - 5, 80)

        for _ in range(n_perturb):
            if used >= self.budget - 3 or not time_ok(5.0):
                break

            # Pick a non-dominated point
            if nd_X:
                idx = np.random.randint(len(nd_X))
                x_seed = nd_X[idx].copy()
            else:
                x_seed = lo + span * np.random.random(self.dim)

            # Gaussian perturbation with decreasing scale
            scale = 0.03 + 0.04 * (1.0 - used / max(self.budget, 1))
            x_pert = x_seed + scale * span * np.random.standard_normal(self.dim)
            x_pert = np.clip(x_pert, lo, hi)
            eval_point(x_pert)

            # Check if we improved the non-dominated set
            if len(all_F) % 10 == 0:
                nd_idx = get_nondominated(all_X, all_F)
                nd_X = [all_X[i] for i in nd_idx]
                nd_F = [all_F[i] for i in nd_idx]

        # Phase 4: Final polish - tight local search on best non-dominated points
        nd_idx = get_nondominated(all_X, all_F)
        nd_X = [all_X[i] for i in nd_idx]
        nd_F = [all_F[i] for i in nd_idx]

        if nd_X:
            # Polish top non-dominated points with very tight local search
            n_polish = min(15, self.budget - used - 2)
            for _ in range(n_polish):
                if used >= self.budget - 2 or not time_ok(5.0):
                    break

                idx = np.random.randint(len(nd_X))
                x_seed = nd_X[idx].copy()

                # Small perturbation
                x_start = x_seed + 0.01 * span * np.random.standard_normal(self.dim)
                x_start = np.clip(x_start, lo, hi)

                # Random weight for scalarization
                w = np.random.dirichlet(np.ones(self.n_obj))

                def scalarized_polish(x, w=w):
                    f = eval_point(x)
                    return float(np.dot(w, f))

                try:
                    res = minimize(
                        scalarized_polish,
                        x_start,
                        method='Nelder-Mead',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 5, 'xatol': 1e-5, 'fatol': 1e-6, 'maxfev': 6}
                    )
                    if used < self.budget and time_ok(3.0):
                        x_res = np.clip(res.x, lo, hi)
                        eval_point(x_res)
                except Exception:
                    pass

        # Build final result
        if all_X:
            X_final = np.array(all_X)
            F_final = np.array(all_F)
        else:
            X_final = np.array([])
            F_final = np.array([])

        return X_final

# EVOLVE-BLOCK-END
