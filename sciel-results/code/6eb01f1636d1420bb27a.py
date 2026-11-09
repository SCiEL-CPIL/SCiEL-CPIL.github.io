"""Multi-Start Nelder-Mead optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Nelder-Mead with adaptive step-size for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, alpha, beta, gamma, max_iter, deadline):
        """
        Deterministic Nelder-Mead simplex method.
        Returns (best_x, best_f, evals_used).
        """
        dim = self.dim
        # Initialize simplex: x0 plus dim perturbed vertices
        simplex = [np.array(x0, dtype=float)]
        for i in range(dim):
            v = np.array(x0, dtype=float)
            v[i] += alpha
            simplex.append(v)

        # Evaluate all vertices
        values = [func(np.clip(v, -5, 5)) for v in simplex]
        evals = dim + 1

        best_x = simplex[np.argmin(values)].copy()
        best_f = min(values)

        for it in range(max_iter):
            if time.time() > deadline:
                break

            # Sort vertices by function value
            order = np.argsort(values)
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]

            # Check convergence: if best is good enough or simplex is tiny
            if values[0] - values[-1] < 1e-10:
                break
            if np.linalg.norm(simplex[0] - simplex[-1]) < 1e-8:
                break

            # Centroid of all but worst vertex
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + gamma * (centroid - simplex[-1])
            fr = func(np.clip(xr, -5, 5))
            evals += 1

            if fr < values[0]:
                # Expansion
                xe = centroid + beta * (xr - centroid)
                fe = func(np.clip(xe, -5, 5))
                evals += 1
                if fe < fr:
                    simplex[-1] = xe
                    values[-1] = fe
                else:
                    simplex[-1] = xr
                    values[-1] = fr
            elif fr < values[-2]:
                simplex[-1] = xr
                values[-1] = fr
            else:
                # Contraction
                if fr < values[-1]:
                    # Outside contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fc = func(np.clip(xc, -5, 5))
                    evals += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        values[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            values[i] = func(np.clip(simplex[i], -5, 5))
                            evals += 1
                else:
                    # Inside contraction
                    xc = centroid - 0.5 * (centroid - simplex[-1])
                    fc = func(np.clip(xc, -5, 5))
                    evals += 1
                    if fc < values[-1]:
                        simplex[-1] = xc
                        values[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            values[i] = func(np.clip(simplex[i], -5, 5))
                            evals += 1

            # Update best
            if values[0] < best_f:
                best_f = values[0]
                best_x = simplex[0].copy()

            # Adaptive alpha: shrink if stuck
            if it > 0 and it % 50 == 0:
                alpha *= 0.5

        return best_x, best_f, evals

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Phase 1: Coarse random search to identify promising basins
        n_init = min(20, budget // 5)
        init_points = [np.zeros(dim)]
        # Add structured points
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-4, 4, dim))

        init_vals = [func(np.clip(p, -5, 5)) for p in init_points]
        evals = n_init

        # Sort by function value
        sorted_idx = np.argsort(init_vals)
        # Take top k starting points for Nelder-Mead
        k_starts = min(5, len(init_vals))
        start_points = [init_points[sorted_idx[i]] for i in range(k_starts)]
        start_vals = [init_vals[sorted_idx[i]] for i in range(k_starts)]

        best_x = start_points[0].copy()
        best_f = start_vals[0]

        # Phase 2: Multi-Start Nelder-Mead
        remaining_budget = budget - evals
        # Allocate budget to each start point, with more budget for better starts
        # Use a weighted allocation
        weights = np.array([1.0 / (i + 1) for i in range(k_starts)])
        weights /= np.sum(weights)
        budgets = (weights * remaining_budget).astype(int)
        # Ensure minimum budget per start
        for i in range(k_starts):
            budgets[i] = max(20, budgets[i])
        # Adjust so total doesn't exceed remaining
        total_alloc = np.sum(budgets)
        if total_alloc > remaining_budget:
            scale = remaining_budget / total_alloc
            budgets = (budgets * scale).astype(int)

        alpha = 0.5  # initial perturbation size
        beta = 2.0   # expansion factor
        gamma = 1.0  # reflection factor

        for i, sp in enumerate(start_points):
            if time.time() > deadline:
                break
            if evals >= budget:
                break

            # Adaptive alpha based on the spread of good points
            # Use the distance between the best and second-best init points
            if i == 0:
                if k_starts > 1:
                    dist = np.linalg.norm(start_points[0] - start_points[1])
                    alpha = max(0.1, min(1.0, dist * 0.5))
                else:
                    alpha = 0.5

            max_iter = budgets[i] // (dim + 1)  # rough estimate
            max_iter = max(10, min(max_iter, 100))

            new_best_x, new_best_f, evals_used = self._nelder_mead(
                func, sp, alpha, beta, gamma, max_iter, deadline
            )
            evals += evals_used

            if new_best_f < best_f:
                best_f = new_best_f
                best_x = new_best_x.copy()

            # If we found a significantly better point, restart NM from there
            # with a smaller alpha
            if i < k_starts - 1 and new_best_f < start_vals[0] - 1.0:
                start_points[i + 1] = new_best_x.copy()
                alpha *= 0.5

        # Phase 3: Final polish with a very small alpha
        if evals < budget and time.time() < deadline:
            remaining = budget - evals
            if remaining > 20:
                polish_budget = min(remaining, 50)
                # Use a very small alpha for final refinement
                new_best_x, new_best_f, evals_used = self._nelder_mead(
                    func, best_x, 0.01, beta, gamma, polish_budget // (dim + 1), deadline
                )
                evals += evals_used
                if new_best_f < best_f:
                    best_f = new_best_f
                    best_x = new_best_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
