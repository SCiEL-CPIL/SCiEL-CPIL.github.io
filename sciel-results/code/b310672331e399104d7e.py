"""Multi-objective optimizer for WFG7 with LHS + Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic LHS sampling with Nelder-Mead local refinement for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use a fixed seed for reproducibility within the run
        rng = np.random.default_rng(42)

        # Phase 1: Latin Hypercube Sampling for diverse initial spread
        # Allocate ~40% of budget to LHS sampling
        lhs_count = min(80, max(20, budget // 3))
        used = 0

        # Generate LHS samples
        X_lhs = np.zeros((lhs_count, dim))
        for d in range(dim):
            # Create LHS points for dimension d
            points = np.linspace(0, 1, lhs_count, endpoint=False)
            points += rng.random(lhs_count) / lhs_count
            points = np.clip(points, 0, 1)
            rng.shuffle(points)
            X_lhs[:, d] = lo[d] + (hi[d] - lo[d]) * points

        # Evaluate LHS samples
        F_lhs = np.zeros((lhs_count, n_obj))
        for i in range(lhs_count):
            if used >= budget:
                break
            F_lhs[i] = problem(X_lhs[i])
            used += 1

        # Phase 2: Nelder-Mead refinement on non-dominated points
        # Find non-dominated points
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # Get non-dominated indices
        nondom_idx = []
        for i in range(lhs_count):
            is_nondom = True
            for j in range(lhs_count):
                if i != j and dominates(F_lhs[j], F_lhs[i]):
                    is_nondom = False
                    break
            if is_nondom:
                nondom_idx.append(i)

        # Sort non-dominated points by sum of objectives (ascending)
        if len(nondom_idx) > 0:
            sums = np.array([np.sum(F_lhs[i]) for i in nondom_idx])
            order = np.argsort(sums)
            nondom_idx = [nondom_idx[i] for i in order]

        # Apply Nelder-Mead to top non-dominated points
        # Allocate remaining budget for refinement
        remaining_budget = budget - used
        # Each Nelder-Mead iteration uses ~dim+1 evaluations
        # Limit number of refinement points based on budget
        max_refine_points = min(len(nondom_idx), max(3, remaining_budget // (dim * 4)))

        def nelder_mead_minimize(objective, x0, bounds, max_iter=20):
            """Simple Nelder-Mead implementation with bounds."""
            dim = len(x0)
            # Initialize simplex
            simplex = [x0.copy() + 1e-4 * (hi - lo)]
            for i in range(dim):
                x = x0.copy()
                x[i] += 1e-4 * (hi[i] - lo[i])
                simplex.append(x)

            f_values = [objective(x) for x in simplex]
            simplex = np.array(simplex)
            f_values = np.array(f_values)

            for iteration in range(max_iter):
                # Sort by objective value
                order = np.argsort(f_values)
                simplex = simplex[order]
                f_values = f_values[order]

                # Compute centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, lo, hi)
                f_reflect = objective(x_reflect)

                if f_values[0] < f_reflect < f_values[-1]:
                    simplex[-1] = x_reflect
                    f_values[-1] = f_reflect
                elif f_reflect < f_values[0]:
                    # Expansion
                    gamma = 2.0
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, lo, hi)
                    f_expand = objective(x_expand)
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        f_values[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        f_values[-1] = f_reflect
                else:
                    # Contraction
                    beta = 0.5
                    if f_reflect < f_values[-1]:
                        x_contract = centroid + beta * (x_reflect - centroid)
                    else:
                        x_contract = centroid - beta * (centroid - simplex[-1])
                    x_contract = np.clip(x_contract, lo, hi)
                    f_contract = objective(x_contract)
                    if f_contract < max(f_reflect, f_values[-1]):
                        simplex[-1] = x_contract
                        f_values[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_values[i] = objective(simplex[i])

                # Check convergence
                if np.max(np.abs(f_values - f_values[0])) < 1e-10:
                    break

            best_idx = np.argmin(f_values)
            return simplex[best_idx], f_values[best_idx]

        # Apply Nelder-Mead to selected non-dominated points
        for idx in nondom_idx[:max_refine_points]:
            if used >= budget:
                break
            if time.time() > deadline:
                break

            x0 = X_lhs[idx].copy()

            # Create scalarized objective (sum of objectives)
            def scalarized(x):
                return float(np.sum(problem(x)))

            # Estimate remaining budget for this refinement
            remaining = budget - used
            max_iter = max(3, min(15, remaining // (dim + 1)))

            x_opt, f_opt = nelder_mead_minimize(scalarized, x0, (lo, hi), max_iter)

            # Update if improved
            if np.sum(f_opt) < np.sum(F_lhs[idx]):
                X_lhs[idx] = x_opt
                F_lhs[idx] = f_opt

            # Update used count (approximate)
            used += max_iter * (dim + 1)

        # Phase 3: If budget remains, do additional random sampling + refinement
        remaining = budget - used
        if remaining > 0 and time.time() < deadline - 30:
            # Get current non-dominated set
            nondom_idx = []
            for i in range(lhs_count):
                is_nondom = True
                for j in range(lhs_count):
                    if i != j and dominates(F_lhs[j], F_lhs[i]):
                        is_nondom = False
                        break
                if is_nondom:
                    nondom_idx.append(i)

            # Do some random perturbations around best points
            for _ in range(min(remaining, 30)):
                if time.time() > deadline:
                    break
                if used >= budget:
                    break

                idx = nondom_idx[rng.integers(len(nondom_idx))]
                x = X_lhs[idx].copy()
                step = 0.02 * (hi - lo)
                x += rng.standard_normal(dim) * step
                x = np.clip(x, lo, hi)
                f = problem(x)
                used += 1

                # Replace if better
                if np.sum(f) < np.sum(F_lhs[idx]):
                    X_lhs[idx] = x
                    F_lhs[idx] = f

        return X_lhs
# EVOLVE-BLOCK-END
