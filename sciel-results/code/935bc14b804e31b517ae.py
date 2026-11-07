"""Multi-objective optimizer on WFG7: Tchebycheff decomposition with Nelder-Mead local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Decomposition-based MOO using Tchebycheff scalarization and Nelder-Mead."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _nelder_mead(self, problem, x0, bounds, weights, max_evals):
        """Run Nelder-Mead on Tchebycheff scalarization. Returns (best_x, best_f, evals_used)."""
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Initialize simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0.copy()
        for i in range(dim):
            simplex[i + 1] = x0.copy()
            step = 0.05 * (hi[i] - lo[i])
            if simplex[i + 1][i] + step <= hi[i]:
                simplex[i + 1][i] += step
            else:
                simplex[i + 1][i] -= step

        def tchebycheff(x):
            fx = problem(x)
            return np.max(weights * (fx - np.min(fx)))  # This won't work correctly

        # Actually, Tchebycheff scalarization: min max_i (w_i * (f_i(x) - z_i*))
        # We need an ideal point estimate. Let's use a running ideal.
        # For simplicity, use the minimum observed so far, or just use f_i directly with weights.
        # Better: min max_i (w_i * f_i(x)) after normalizing, or use the standard form.
        # Let's use: min max_i (w_i * (f_i(x) - ref_i)) where ref is a reference.
        # Or simpler: min max_i (w_i * f_i(x)) - this biases toward lower f_i.
        # Standard Tchebycheff: min_x max_i (w_i * (f_i(x) - z_i*))
        # We'll use z_i* = 0 as a rough ideal, or better, track the best seen.
        # For this implementation, let's use: min max_i (w_i * f_i(x))
        # This is a valid scalarization for non-dominated solutions when weights are positive.

        def scalarize(x):
            fx = problem(x)
            return np.max(weights * fx)

        # Evaluate initial simplex
        fvals = np.zeros(dim + 1)
        evals = 0
        for i in range(dim + 1):
            simplex[i] = np.clip(simplex[i], lo, hi)
            fvals[i] = scalarize(simplex[i])
            evals += 1

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        max_iter = max_evals - (dim + 1)

        for _ in range(max_iter):
            if evals >= max_evals:
                break

            # Sort by objective value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, lo, hi)
            fr = scalarize(xr)
            evals += 1

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, lo, hi)
                fe = scalarize(xe)
                evals += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                xc = np.clip(xc, lo, hi)
                fc = scalarize(xc)
                evals += 1

                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lo, hi)
                        fvals[i] = scalarize(simplex[i])
                        evals += 1
                        if evals >= max_evals:
                            break
                    if evals >= max_evals:
                        break

        # Find best point
        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        best_f = problem(best_x)
        evals += 1  # This is the final evaluation

        return best_x, best_f, evals

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Generate uniformly distributed weight vectors for 3 objectives
        # Using Dirichlet distribution for uniform spread on the simplex
        n_weights = 30
        weights = np.zeros((n_weights, n_obj))
        # Use a grid-like approach for 3D: sum of weights = 1
        # For 3 objectives, we can use a triangular grid
        n_per_axis = 6  # 6 values along each axis, giving 6*5/2 = 15 unique + boundaries
        # Actually, let's use a proper uniform sampling
        for i in range(n_weights):
            # Sample from Dirichlet(1,1,1) which is uniform on the simplex
            w = np.random.dirichlet([1.0, 1.0, 1.0])
            weights[i] = w

        # Ensure we have a good spread by adding some canonical weights
        canonical = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [1/3, 1/3, 1/3],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        # Replace first 10 weights with canonical, keep rest as Dirichlet samples
        weights[:10] = canonical
        # Resample remaining with more spread
        for i in range(10, n_weights):
            weights[i] = np.random.dirichlet([1.0, 1.0, 1.0])

        # Initial population: center + random
        center = (lo + hi) / 2.0
        X = np.zeros((n_weights, dim))
        F = np.zeros((n_weights, n_obj))
        used = 0

        # Initialize each subproblem with a random point near center
        for i in range(n_weights):
            if time.time() > deadline:
                break
            # Start near center with some randomness
            x_init = center + 0.1 * (hi - lo) * np.random.randn(dim)
            x_init = np.clip(x_init, lo, hi)
            X[i] = x_init
            F[i] = problem(x_init)
            used += 1

        # Allocate remaining budget to subproblems
        remaining = budget - used
        if remaining <= 0:
            return X

        # Each subproblem gets roughly equal budget
        budget_per_sub = max(1, remaining // n_weights)

        # Run Nelder-Mead for each subproblem
        for i in range(n_weights):
            if used >= budget:
                break
            if time.time() > deadline:
                break

            # Allocate budget for this subproblem
            alloc = min(budget_per_sub, budget - used)
            if alloc < 2:
                break

            # Run local search
            best_x, best_f, evals_used = self._nelder_mead(
                problem, X[i], bounds, weights[i], alloc
            )
            used += evals_used

            # Update
            X[i] = best_x
            F[i] = best_f

            # If we still have budget, do a second round with smaller perturbations
            if used < budget and time.time() < deadline:
                alloc2 = min(budget_per_sub // 2, budget - used)
                if alloc2 >= 2:
                    best_x2, best_f2, evals_used2 = self._nelder_mead(
                        problem, X[i], bounds, weights[i], alloc2
                    )
                    used += evals_used2
                    # Keep the better result
                    if np.any(best_f2 < F[i]) and not np.any(best_f2 > F[i]):
                        X[i] = best_x2
                        F[i] = best_f2
                    elif np.sum(best_f2) < np.sum(F[i]):
                        X[i] = best_x2
                        F[i] = best_f2

        # Final polish: for each non-dominated point, do a small local search
        if used < budget and time.time() < deadline:
            # Find non-dominated points
            non_dom_mask = np.ones(n_weights, dtype=bool)
            for i in range(n_weights):
                for j in range(n_weights):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        non_dom_mask[i] = False
                        break

            # Polish non-dominated points
            for i in np.where(non_dom_mask)[0]:
                if used >= budget or time.time() > deadline:
                    break
                # Small random perturbations
                for _ in range(3):
                    if used >= budget or time.time() > deadline:
                        break
                    trial = X[i] + 0.01 * (hi - lo) * np.random.randn(dim)
                    trial = np.clip(trial, lo, hi)
                    trial_f = problem(trial)
                    used += 1
                    if np.all(trial_f <= F[i]) and np.any(trial_f < F[i]):
                        X[i] = trial
                        F[i] = trial_f

        return X
# EVOLVE-BLOCK-END
