"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive step sizing for Schaffers F7."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Track best found
        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = float(func(x))
            call_count += 1
            return val

        # Deterministic multi-start points in [-4, 4]^5
        # Use a mix of structured and pseudo-random points
        rng = np.random.default_rng(42)

        # Generate 12 starting points
        n_starts = 12
        starts = []

        # Point 1: origin
        starts.append(np.zeros(dim))

        # Points 2-5: axis-aligned extremes
        for axis in range(min(4, dim)):
            p = np.zeros(dim)
            p[axis] = 3.0
            starts.append(p)

        # Points 6-12: random points in [-4, 4]^dim
        for _ in range(n_starts - 5):
            starts.append(rng.uniform(-4.0, 4.0, dim))

        # Shuffle starts with a fixed seed for reproducibility
        perm = rng.permutation(n_starts)
        starts = [starts[i] for i in perm]

        # Budget allocation: first start gets more budget, rest get less
        # 500 total calls. Reserve 1 for initial evaluation of first start.
        # Give first 2 starts 80 calls each, remaining 10 starts 30 calls each
        # 2*80 + 10*30 = 160 + 300 = 460, leaving 40 for refinement
        # Actually let's be more aggressive: 12 starts * ~40 calls = 480, plus refinement
        # Let's do: 12 starts with 35 calls each = 420, then 80 for final refinement

        calls_per_start = 35
        refinement_budget = budget - n_starts * calls_per_start
        if refinement_budget < 20:
            calls_per_start = max(10, (budget - 20) // n_starts)
            refinement_budget = budget - n_starts * calls_per_start

        best_starts = []  # (f, x) for top starts

        for s_idx, x0 in enumerate(starts):
            # Check deadline
            if time.time() > deadline - 5:
                break
            if call_count >= budget:
                break

            # Nelder-Mead for this start
            x = np.array(x0, dtype=np.float64)
            f = safe_call(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

            # Initialize simplex
            simplex = [x.copy()]
            simplex_f = [f]

            # Add perturbation points
            initial_step = 1.0
            for i in range(dim):
                xi = x.copy()
                xi[i] += initial_step
                simplex.append(xi.copy())
                fi = safe_call(xi)
                simplex_f.append(fi)

            # Sort simplex
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            # Track local best for this start
            local_best_f = min(simplex_f)
            local_best_x = simplex[0].copy()

            # Nelder-Mead iterations
            max_nm_iter = calls_per_start - (1 + dim)  # minus initial simplex evaluations
            if max_nm_iter < 1:
                max_nm_iter = 1

            for it in range(max_nm_iter):
                if time.time() > deadline - 5:
                    break
                if call_count >= budget:
                    break

                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = safe_call(x_ref)

                if f_ref < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex[-1])
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
                    rho = 0.5
                    if f_ref < simplex_f[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (x_ref - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = safe_call(x_con)

                    if f_con < min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex_f[i] = safe_call(simplex[i])

                # Re-sort
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Update local best
                if simplex_f[0] < local_best_f:
                    local_best_f = simplex_f[0]
                    local_best_x = simplex[0].copy()

                # Update global best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()

                # Check convergence
                if len(simplex_f) > 1:
                    f_spread = simplex_f[-1] - simplex_f[0]
                    if f_spread < 1e-10:
                        break

            best_starts.append((local_best_f, local_best_x))

        # Refinement phase: run Nelder-Mead from the best start with remaining budget
        if call_count < budget and best_x is not None and refinement_budget > 10:
            if time.time() <= deadline - 5:
                x = best_x.copy()
                f = safe_call(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

                # Initialize simplex around best
                simplex = [x.copy()]
                simplex_f = [f]

                step = 0.5
                for i in range(dim):
                    xi = x.copy()
                    xi[i] += step
                    simplex.append(xi.copy())
                    fi = safe_call(xi)
                    simplex_f.append(fi)

                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                max_refine_iter = refinement_budget - (1 + dim)
                if max_refine_iter < 1:
                    max_refine_iter = 1

                for it in range(max_refine_iter):
                    if time.time() > deadline - 2:
                        break
                    if call_count >= budget:
                        break

                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    alpha = 1.0
                    x_ref = centroid + alpha * (centroid - simplex[-1])
                    x_ref = np.clip(x_ref, -5.0, 5.0)
                    f_ref = safe_call(x_ref)

                    if f_ref < simplex_f[0]:
                        gamma = 2.0
                        x_exp = centroid + gamma * (centroid - simplex[-1])
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
                        rho = 0.5
                        if f_ref < simplex_f[-1]:
                            x_con = centroid + rho * (x_ref - centroid)
                        else:
                            x_con = centroid - rho * (x_ref - centroid)
                        x_con = np.clip(x_con, -5.0, 5.0)
                        f_con = safe_call(x_con)

                        if f_con < min(f_ref, simplex_f[-1]):
                            simplex[-1] = x_con
                            simplex_f[-1] = f_con
                        else:
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex_f[i] = safe_call(simplex[i])

                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]

                    if simplex_f[0] < best_f:
                        best_f = simplex_f[0]
                        best_x = simplex[0].copy()

                    if len(simplex_f) > 1:
                        f_spread = simplex_f[-1] - simplex_f[0]
                        if f_spread < 1e-12:
                            break

        # If we still have budget, do a final local search from best
        if call_count < budget and best_x is not None and time.time() <= deadline - 2:
            x = best_x.copy()
            f = best_f
            sigma = 0.3
            for _ in range(budget - call_count):
                if time.time() > deadline - 1:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy < f:
                    x, f = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.9
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
