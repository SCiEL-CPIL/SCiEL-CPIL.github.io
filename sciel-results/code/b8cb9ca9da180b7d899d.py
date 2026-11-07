"""Rewritten program: adaptive multi-start L-BFGS-B multi-objective optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools


class Optimizer:
    """Multi-objective optimizer using adaptive multi-start L-BFGS-B with non-dominated sorting."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Track all evaluated points and their objective values
        all_X = []
        all_F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped), dtype=float)
            eval_count += 1
            all_X.append(x_clipped.copy())
            all_F.append(f)
            return f

        # Helper: non-dominated sorting
        def is_dominated(f1, F_set):
            """Check if f1 is dominated by any in F_set."""
            for f2 in F_set:
                if np.all(f2 <= f1) and np.any(f2 < f1):
                    return True
            return False

        def get_pareto_front(F_arr):
            """Return indices of non-dominated points."""
            n = len(F_arr)
            nondom = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    nondom.append(i)
            return nondom

        # Generate diverse starting points using Latin Hypercube-like sampling
        def lhs_sample(n_samples, dim):
            """Generate n_samples points in [0,1]^dim using LHS."""
            points = np.zeros((n_samples, dim))
            for d in range(dim):
                perm = np.arange(n_samples)
                np.random.shuffle(perm)
                points[:, d] = (perm + 0.5) / n_samples
            return points

        # Phase 1: Initial diverse sampling
        n_initial = min(30, budget // 5)
        lhs_pts = lhs_sample(n_initial, dim)
        for pt in lhs_pts:
            x = lo + span * pt
            evaluate(x)
            if eval_count >= budget:
                break

        # Phase 2: Multi-start L-BFGS-B on various objective combinations
        # Use weighted sum and augmented Tchebycheff approaches
        # Generate weight vectors
        n_weight_starts = min(20, max(5, (budget - eval_count) // 15))

        # Create diverse weight vectors
        weight_vectors = []
        # Uniform weights
        weight_vectors.append(np.ones(n_obj) / n_obj)
        # Single-objective weights
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weight_vectors.append(w)
        # Random weights
        rng_local = np.random.RandomState(42)
        for _ in range(n_weight_starts - 2):
            w = rng_local.dirichlet(np.ones(n_obj))
            weight_vectors.append(w)

        # Take starting points from current Pareto front and diverse points
        F_arr = np.array(all_F)
        X_arr = np.array(all_X)

        # Get current Pareto front indices
        pareto_idx = get_pareto_front(F_arr)
        # Also add some diverse random points as starts
        start_candidates = [X_arr[i] for i in pareto_idx]
        # Add a few random points
        for _ in range(min(5, len(X_arr))):
            idx = rng_local.randint(len(X_arr))
            start_candidates.append(X_arr[idx])

        # For each weight vector, run L-BFGS-B from best starting point
        for w in weight_vectors:
            if eval_count >= budget:
                break

            # Select best starting point for this weight
            # Score starts by weighted objective
            best_start = None
            best_score = np.inf
            for sx in start_candidates:
                f = np.array(problem(np.clip(sx, lo, hi)), dtype=float)
                eval_count += 1
                all_X.append(np.clip(sx, lo, hi).copy())
                all_F.append(f)
                score = np.dot(w, f)
                if score < best_score:
                    best_score = score
                    best_start = sx.copy()

            if eval_count >= budget or best_start is None:
                continue

            # Run L-BFGS-B to minimize weighted sum
            def weighted_obj(x, w=w):
                x_c = np.clip(x, lo, hi)
                f = np.array(problem(x_c), dtype=float)
                eval_count += 1
                all_X.append(x_c.copy())
                all_F.append(f)
                return np.dot(w, f)

            bounds_list = list(zip(lo, hi))
            try:
                res = minimize(
                    weighted_obj,
                    best_start,
                    method='L-BFGS-B',
                    bounds=bounds_list,
                    options={'maxiter': 50, 'maxfun': 20}
                )
            except Exception:
                pass

            if eval_count >= budget:
                break

            # Also try augmented Tchebycheff for this weight
            if eval_count < budget:
                def tcheby_obj(x, w=w):
                    x_c = np.clip(x, lo, hi)
                    f = np.array(problem(x_c), dtype=float)
                    eval_count += 1
                    all_X.append(x_c.copy())
                    all_F.append(f)
                    # Augmented Tchebycheff: max_i w_i * |f_i - z_i| + rho * sum(w_i * |f_i - z_i|)
                    # Use current best as reference
                    F_curr = np.array(all_F)
                    z = np.min(F_curr, axis=0)
                    terms = w * np.abs(f - z)
                    return np.max(terms) + 0.01 * np.sum(terms)

                try:
                    res2 = minimize(
                        tcheby_obj,
                        best_start,
                        method='L-BFGS-B',
                        bounds=bounds_list,
                        options={'maxiter': 30, 'maxfun': 15}
                    )
                except Exception:
                    pass

        # Phase 3: Local refinement around best points
        if eval_count < budget:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)
            pareto_idx = get_pareto_front(F_arr)

            # Perturb and refine around Pareto points
            for idx in pareto_idx[:10]:
                if eval_count >= budget:
                    break
                base = X_arr[idx].copy()
                # Try small perturbations
                for scale in [0.05, 0.1, 0.2]:
                    if eval_count >= budget:
                        break
                    for _ in range(3):
                        if eval_count >= budget:
                            break
                        perturb = base + scale * span * np.random.randn(dim)
                        x_c = np.clip(perturb, lo, hi)
                        f = np.array(problem(x_c), dtype=float)
                        eval_count += 1
                        all_X.append(x_c.copy())
                        all_F.append(f)

        # Phase 4: Final sweep - fill remaining budget with smart random sampling
        if eval_count < budget:
            F_arr = np.array(all_F)
            # Find regions with high density of good solutions
            # Sample around the best points
            for _ in range(budget - eval_count):
                if eval_count >= budget:
                    break
                # Gaussian perturbation around random existing point
                idx = np.random.randint(len(all_X))
                base = all_X[idx]
                sigma = 0.1 * span
                x_c = np.clip(base + np.random.randn(dim) * sigma, lo, hi)
                f = np.array(problem(x_c), dtype=float)
                eval_count += 1
                all_X.append(x_c.copy())
                all_F.append(f)

        return np.array(all_X)
# EVOLVE-BLOCK-END
