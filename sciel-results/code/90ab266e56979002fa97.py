"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive reset for ill-conditioned landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin from 330s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls_used = 0
        best_fx = float('inf')
        best_x = np.zeros(dim)

        def check_time():
            return time.time() < deadline - 5.0

        def check_budget():
            return calls_used < budget

        def nm_step(simplex, simplex_f, shrink=0.5, expand=2.0):
            """One Nelder-Mead iteration. Returns updated simplex, simplex_f, and calls_made."""
            nonlocal calls_used, best_fx, best_x
            calls_made = 0

            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]

            # Update global best
            if simplex_f[0] < best_fx:
                best_fx = simplex_f[0]
                best_x = simplex[0].copy()

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            worst = simplex[-1]
            reflected = centroid + expand * (centroid - worst)
            reflected = np.clip(reflected, -5, 5)

            if not check_budget() or not check_time():
                return simplex, simplex_f, calls_made

            f_reflected = func(reflected)
            calls_used += 1
            calls_made += 1

            if f_reflected < simplex_f[0]:
                # Expansion
                if not check_budget() or not check_time():
                    simplex[-1] = reflected
                    simplex_f[-1] = f_reflected
                    return simplex, simplex_f, calls_made

                expanded = centroid + 2.0 * (reflected - centroid)
                expanded = np.clip(expanded, -5, 5)
                f_expanded = func(expanded)
                calls_used += 1
                calls_made += 1

                if f_expanded < f_reflected:
                    simplex[-1] = expanded
                    simplex_f[-1] = f_expanded
                else:
                    simplex[-1] = reflected
                    simplex_f[-1] = f_reflected

            elif f_reflected < simplex_f[-2]:
                # Simple reflection
                simplex[-1] = reflected
                simplex_f[-1] = f_reflected

            else:
                # Contraction
                if f_reflected < simplex_f[-1]:
                    # Outside contraction
                    contracted = centroid + 0.5 * (reflected - centroid)
                    f_contracted_type = 'outside'
                else:
                    # Inside contraction
                    contracted = centroid + 0.5 * (worst - centroid)
                    f_contracted_type = 'inside'

                contracted = np.clip(contracted, -5, 5)

                if not check_budget() or not check_time():
                    return simplex, simplex_f, calls_made

                f_contracted = func(contracted)
                calls_used += 1
                calls_made += 1

                if f_contracted < simplex_f[-1]:
                    simplex[-1] = contracted
                    simplex_f[-1] = f_contracted
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        if not check_budget() or not check_time():
                            break
                        simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = func(simplex[i])
                        calls_used += 1
                        calls_made += 1
                        if simplex_f[i] < best_fx:
                            best_fx = simplex_f[i]
                            best_x = simplex[i].copy()

            return simplex, simplex_f, calls_made

        def run_nelder_mead(x0, sigma0, max_iter=200):
            """Run Nelder-Mead from initial point x0 with initial step sigma0."""
            nonlocal calls_used, best_fx, best_x

            if not check_budget() or not check_time():
                return

            # Build initial simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(dim):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += sigma0
                simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)

            # Evaluate simplex
            simplex_f = np.zeros(dim + 1)
            for i in range(dim + 1):
                if not check_budget() or not check_time():
                    return
                simplex_f[i] = func(simplex[i])
                calls_used += 1
                if simplex_f[i] < best_fx:
                    best_fx = simplex_f[i]
                    best_x = simplex[i].copy()

            # Main loop
            for _ in range(max_iter):
                if not check_budget() or not check_time():
                    break

                simplex, simplex_f, _ = nm_step(simplex, simplex_f)

                # Check convergence
                f_range = np.max(simplex_f) - np.min(simplex_f)
                if f_range < 1e-10:
                    break

            return

        # Phase 1: Multi-start with diverse initial points
        # Use a mix of random points and points sampled from a grid
        n_starts = 5
        initial_points = []

        # First start: center
        initial_points.append(np.zeros(dim))

        # Remaining starts: random points
        for _ in range(n_starts - 1):
            x = rng.uniform(-4, 4, dim)
            initial_points.append(x)

        # Evaluate all initial points first to find a good starting region
        for x0 in initial_points:
            if not check_budget() or not check_time():
                break
            fx = func(x0)
            calls_used += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x0.copy()

        # Sort initial points by fitness
        initial_points.sort(key=lambda x: func(x) if False else 0)  # Already evaluated

        # Phase 2: Run Nelder-Mead from each start point, prioritizing better ones
        # We'll run them in order of initial fitness, but with decreasing max_iter
        # to allocate more budget to promising starts

        # Re-evaluate to get fitness values (we already have them in best tracking,
        # but let's be explicit)
        start_fitnesses = []
        for x0 in initial_points:
            if not check_budget() or not check_time():
                break
            # We already evaluated these, so use cached values
            # For simplicity, just proceed with Nelder-Mead
            start_fitnesses.append(best_fx)  # Approximate

        # Run Nelder-Mead from best start first, then others
        # Allocate budget proportionally: first start gets 40%, rest share 60%
        remaining_budget = budget - calls_used
        if remaining_budget <= 0:
            return

        # Adaptive: run NM from each start, with max_iter based on remaining budget
        # First start gets more iterations
        max_iters = [int(remaining_budget * 0.4 // (dim + 1)),
                     int(remaining_budget * 0.15 // (dim + 1)),
                     int(remaining_budget * 0.15 // (dim + 1)),
                     int(remaining_budget * 0.15 // (dim + 1)),
                     int(remaining_budget * 0.15 // (dim + 1))]

        # Ensure at least some iterations
        max_iters = [max(10, mi) for mi in max_iters]

        # Initial sigma for Nelder-Mead: adapt based on problem scale
        base_sigma = 1.5

        for i, x0 in enumerate(initial_points):
            if not check_budget() or not check_time():
                break

            # Use the best point found so far as a potential refinement start
            # for later starts, to leverage progress
            if i == 0:
                x_start = x0.copy()
                sigma_start = base_sigma
            else:
                # For subsequent starts, blend with best found so far
                # This helps if the first start found a good region
                x_start = 0.5 * x0 + 0.5 * best_x
                x_start = np.clip(x_start, -5, 5)
                sigma_start = base_sigma * 0.5  # Smaller step since we have a better reference

            run_nelder_mead(x_start, sigma_start, max_iter=max_iters[i])

        # Phase 3: Final refinement with small steps around best point
        if check_budget() and check_time():
            # Local refinement with very small perturbations
            refine_sigma = 0.1
            for _ in range(min(50, budget - calls_used)):
                if not check_budget() or not check_time():
                    break
                x_try = best_x + refine_sigma * rng.standard_normal(dim)
                x_try = np.clip(x_try, -5, 5)
                f_try = func(x_try)
                calls_used += 1
                if f_try < best_fx:
                    best_fx = f_try
                    best_x = x_try.copy()
                    # Slightly reduce step size on success
                    refine_sigma *= 0.9
                else:
                    refine_sigma *= 0.95

        return best_x, best_fx
# EVOLVE-BLOCK-END
