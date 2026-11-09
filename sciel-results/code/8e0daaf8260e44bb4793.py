"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive Multi-Start Nelder-Mead with LHS exploration and high-precision polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = np.inf

            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead(x0, f0, tight=False):
            """Nelder-Mead with optional tight convergence for final polish."""
            nonlocal calls_made, best_x, best_f
            if time.time() > deadline or calls_made >= budget - 3:
                return

            # Initialize simplex with scaled random directions (handles rotated landscapes)
            simplex = [x0.copy()]
            simplex_f = [f0]

            rng = np.random.default_rng(12345 + calls_made)
            for i in range(dim):
                direction = rng.standard_normal(dim)
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    direction = np.zeros(dim)
                    direction[i] = 1.0
                    norm = 1.0
                direction /= norm
                # Scale: 0.3 for initial, 0.05 for tight polish
                scale = 0.05 if tight else 0.3
                new_x = x0 + scale * direction
                new_x = np.clip(new_x, -5.0, 5.0)
                if np.allclose(new_x, x0):
                    new_x[i] += 0.01 if new_x[i] < 0 else -0.01
                    new_x = np.clip(new_x, -5.0, 5.0)
                f_new = safe_call(new_x)
                simplex.append(new_x)
                simplex_f.append(f_new)

                if time.time() > deadline or calls_made >= budget - 3:
                    return

            # NM Parameters
            alpha = 1.0
            gamma = 2.0
            sigma = 0.5

            # Convergence threshold: tighter for polish
            tol = 1e-12 if tight else 1e-8

            max_iter = 200 if tight else 150
            for _ in range(max_iter):
                if time.time() > deadline or calls_made >= budget - 3:
                    break

                # Sort simplex
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]

                # Check for convergence (both function and spread)
                f_spread = abs(simplex_f[0] - simplex_f[-1])
                x_spread = np.max(np.linalg.norm(np.array(simplex) - np.array(simplex[0]), axis=1))
                if f_spread < tol * (1 + abs(simplex_f[0])) and x_spread < (0.001 if tight else 0.01):
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = safe_call(x_ref)

                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = safe_call(x_exp)

                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < simplex_f[-1]:
                        # Outside contraction
                        x_con = centroid + 0.5 * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - 0.5 * (x_ref - centroid)

                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = safe_call(x_con)

                    if f_con < min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            simplex_f[i] = safe_call(simplex[i])

                # Update global best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()

        def lhs_sample(n_samples, dim, rng):
            """Generate n_samples points via Latin Hypercube Sampling in [-5,5]^dim."""
            points = np.zeros((n_samples, dim))
            for d in range(dim):
                perm = rng.permutation(n_samples)
                points[:, d] = (perm + rng.random(n_samples)) / n_samples
                points[:, d] = points[:, d] * 10.0 - 5.0
            return points

        # Phase 1: Structured Exploration with LHS + key points
        rng = np.random.default_rng(42)

        # Evaluate origin
        x_origin = np.zeros(dim)
        f_origin = safe_call(x_origin)
        best_x = x_origin.copy()
        best_f = f_origin

        # Evaluate axis-aligned extreme points (cheap, catches shifted minima on axes)
        for i in range(dim):
            for sign in [1, -1]:
                x = np.zeros(dim)
                x[i] = sign * 4.0
                if time.time() < deadline and calls_made < budget - 50:
                    f = safe_call(x)
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()

        # LHS sampling: use ~25% of budget for exploration
        explore_budget = int(budget * 0.25)
        n_lhs = explore_budget - calls_made - 1
        if n_lhs > 0 and time.time() < deadline:
            lhs_points = lhs_sample(n_lhs, dim, rng)
            for x in lhs_points:
                if time.time() > deadline or calls_made >= budget - 30:
                    break
                f = safe_call(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

        # Phase 2: Local Refinement from best point
        if time.time() < deadline and calls_made < budget - 10:
            nelder_mead(best_x.copy(), best_f, tight=False)

        # Phase 3: Adaptive Multi-Start with local perturbations
        # Generate perturbations around the current best to find nearby basins
        n_starts = 8
        for i in range(n_starts):
            if time.time() > deadline or calls_made >= budget - 15:
                break

            # Decreasing scale perturbations around best
            scale = 2.0 / (i + 1)
            direction = rng.standard_normal(dim)
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            x_new = best_x + scale * direction
            x_new = np.clip(x_new, -5.0, 5.0)
            f_new = safe_call(x_new)

            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
                if time.time() < deadline and calls_made < budget - 10:
                    nelder_mead(best_x.copy(), best_f, tight=False)

        # Phase 4: High-precision polish on the final best point
        if time.time() < deadline and calls_made < budget - 5:
            nelder_mead(best_x.copy(), best_f, tight=True)

        return best_x, best_f
# EVOLVE-BLOCK-END
