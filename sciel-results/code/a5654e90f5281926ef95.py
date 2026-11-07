import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = [0]  # Mutable counter
        all_X = []
        all_F = []

        def safe_eval(x):
            if used[0] >= self.budget:
                return None
            try:
                x = np.clip(x, lo, hi)
                f = problem(x)
                used[0] += 1
                return f
            except Exception:
                return None

        def add_point(x, f):
            f = np.array(f)
            # Check if dominated
            dominated = False
            for af in all_F:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            if dominated:
                return
            # Remove dominated points
            new_X = []
            new_F = []
            for i, (ax, af) in enumerate(zip(all_X, all_F)):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_X.append(ax)
                    new_F.append(af)
            new_X.append(x)
            new_F.append(f)
            all_X.clear()
            all_F.clear()
            all_X.extend(new_X)
            all_F.extend(new_F)

        # Phase 1: Latin Hypercube Sampling for initial exploration
        n_init = 20
        lhs_points = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.arange(n_init)
            np.random.shuffle(perm)
            lhs_points[:, d] = (perm + 0.5) / n_init

        X_init = lo + lhs_points * (hi - lo)

        for x in X_init:
            if time.time() > deadline or used[0] >= self.budget:
                break
            f = safe_eval(x)
            if f is not None:
                add_point(x, f)

        # Phase 2: Decomposition-based local search (MOEA/D style)
        # Generate weight vectors on the simplex
        # Use a grid of weights for 3 objectives
        weights_list = []
        # Add vertices
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights_list.append(w)

        # Add edge midpoints
        for i in range(self.n_obj):
            for j in range(i + 1, self.n_obj):
                w = np.zeros(self.n_obj)
                w[i] = 0.5
                w[j] = 0.5
                weights_list.append(w)

        # Add center
        w = np.ones(self.n_obj) / self.n_obj
        weights_list.append(w)

        # Add additional structured weights for better coverage
        extra_weights = [
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
            [0.3, 0.5, 0.2],
            [0.4, 0.4, 0.2],
            [0.4, 0.2, 0.4],
            [0.2, 0.4, 0.4],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.5, 0.3, 0.2],
            [0.3, 0.5, 0.2],
            [0.2, 0.3, 0.5],
        ]
        for ew in extra_weights:
            w = np.array(ew)
            w = w / np.sum(w)
            weights_list.append(w)

        weights_list = np.array(weights_list)

        # For each weight vector, run L-BFGS-B from multiple diverse starting points
        # This is the decomposition-based approach: each subproblem is solved independently

        # Determine max evaluations per subproblem
        n_subproblems = len(weights_list)
        n_starts_per_sub = 2  # Multiple starts per subproblem for robustness
        total_subproblem_runs = n_subproblems * n_starts_per_sub
        remaining_budget = self.budget - used[0]
        maxfev_per_run = max(3, remaining_budget // total_subproblem_runs)

        # Get reference points for normalization
        def get_nadir():
            if all_F:
                return np.max(all_F, axis=0)
            return np.array([1705.0, 12.0, 0.3])

        def get_ideal():
            if all_F:
                return np.min(all_F, axis=0)
            return np.zeros(self.n_obj)

        # Collect diverse starting points from the archive
        start_points = []
        if all_X:
            start_points = [x.copy() for x in all_X]
        # Add center
        start_points.append((lo + hi) / 2.0)
        # Add some random points
        for _ in range(5):
            start_points.append(lo + np.random.uniform(0, 1, self.dim) * (hi - lo))

        for wi, w in enumerate(weights_list):
            if time.time() > deadline or used[0] >= self.budget:
                break

            nadir = get_nadir()
            ideal = get_ideal()

            for start_idx in range(n_starts_per_sub):
                if time.time() > deadline or used[0] >= self.budget:
                    break

                # Choose starting point
                if start_idx == 0:
                    # Start from the best point for this weight
                    if all_F:
                        F_arr = np.array(all_F)
                        weighted_sums = F_arr @ w
                        best_idx = np.argmin(weighted_sums)
                        x0 = all_X[best_idx].copy()
                    else:
                        x0 = (lo + hi) / 2.0
                else:
                    # Start from a different point for diversity
                    if start_points:
                        idx = (wi + start_idx) % len(start_points)
                        x0 = start_points[idx].copy()
                        # Add small perturbation
                        x0 = x0 + np.random.uniform(-0.1, 0.1, self.dim)
                        x0 = np.clip(x0, lo, hi)
                    else:
                        x0 = (lo + hi) / 2.0

                def weighted_obj(x, w=w):
                    if used[0] >= self.budget:
                        return 1e10
                    f = problem(x)
                    used[0] += 1
                    return np.dot(f, w)

                try:
                    res = minimize(
                        weighted_obj,
                        x0,
                        method='L-BFGS-B',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 10, 'maxfev': maxfev_per_run, 'ftol': 1e-6}
                    )

                    if used[0] < self.budget:
                        x_best = np.clip(res.x, lo, hi)
                        f_best = safe_eval(x_best)
                        if f_best is not None:
                            add_point(x_best, f_best)

                except Exception:
                    pass

        # Phase 3: Local refinement around non-dominated points
        if all_F and used[0] < self.budget:
            # Get non-dominated points
            nd_X = []
            nd_F = []
            for i, (ax, af) in enumerate(zip(all_X, all_F)):
                dominated = False
                for j, (bx, bf) in enumerate(zip(all_X, all_F)):
                    if i != j and np.all(bf <= af) and np.any(bf < af):
                        dominated = True
                        break
                if not dominated:
                    nd_X.append(ax)
                    nd_F.append(af)

            # Refine each non-dominated point with small local search
            for ax in nd_X[:10]:  # Limit to 10 points
                if time.time() > deadline or used[0] >= self.budget:
                    break

                x0 = ax.copy()

                def sum_obj(x):
                    if used[0] >= self.budget:
                        return 1e10
                    f = problem(x)
                    used[0] += 1
                    return np.sum(f)

                try:
                    res = minimize(
                        sum_obj,
                        x0,
                        method='L-BFGS-B',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 5, 'maxfev': 3, 'ftol': 1e-8, 'gtol': 1e-8}
                    )
                    if used[0] < self.budget:
                        x_best = np.clip(res.x, lo, hi)
                        f_best = safe_eval(x_best)
                        if f_best is not None:
                            add_point(x_best, f_best)
                except Exception:
                    pass

        # Phase 4: If budget remains, do random perturbation around best points
        if used[0] < self.budget and all_X:
            remaining = self.budget - used[0]
            n_perturb = min(remaining, 30)
            step_size = 0.05 * (hi - lo)
            for _ in range(n_perturb):
                if used[0] >= self.budget or time.time() > deadline:
                    break
                idx = np.random.randint(len(all_X))
                x_new = all_X[idx] + step_size * np.random.standard_normal(self.dim)
                x_new = np.clip(x_new, lo, hi)
                f_new = safe_eval(x_new)
                if f_new is not None:
                    add_point(x_new, f_new)

        return all_X
