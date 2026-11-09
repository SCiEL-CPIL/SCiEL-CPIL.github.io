"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Halton global search + coordinate-wise lattice refinement + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Phase 1: Halton sequence global search (60% of budget)
        n_halton = int(budget * 0.6)
        # Phase 2: Coordinate-wise lattice refinement (20% of budget)
        n_lattice = int(budget * 0.2)
        # Phase 3: Nelder-Mead polish (20% of budget)
        n_nm = budget - n_halton - n_lattice

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def try_point(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # --- Phase 1: Halton sequence ---
        # Generate Halton sequence points in [0,1]^dim, map to [-5,5]^dim
        def halton(index, base):
            """Generate the index-th point of the Halton sequence in base."""
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        # We'll use a stride to avoid the very first points (which cluster near origin)
        # and to get good coverage
        stride = 13  # prime stride for better coverage
        for i in range(n_halton):
            if time.time() > deadline:
                break
            idx = (i + 1) * stride
            x = np.zeros(dim)
            for d in range(dim):
                # Use different primes for each dimension
                base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][d % 10]
                x[d] = -5 + 10 * halton(idx, base)
            try_point(x)

        # --- Phase 2: Coordinate-wise lattice refinement around best point ---
        if best_x is not None and calls_made < budget:
            # Do coordinate-wise searches: for each dimension, try a few offsets
            # This is a structured local refinement
            step_sizes = [1.0, 0.5, 0.25, 0.125, 0.0625]
            for step in step_sizes:
                if time.time() > deadline or calls_made >= budget:
                    break
                for d in range(dim):
                    if time.time() > deadline or calls_made >= budget:
                        break
                    for delta in [-step, step]:
                        if calls_made >= budget:
                            break
                        x_new = best_x.copy()
                        x_new[d] += delta
                        try_point(x_new)

        # --- Phase 3: Nelder-Mead polish ---
        if best_x is not None and calls_made < budget and n_nm > 0:
            # Simple Nelder-Mead implementation
            def nm_minimize(func, x0, budget_left, deadline):
                dim = len(x0)
                # Initialize simplex
                simplex = [x0.copy()]
                f0 = func(np.clip(x0, -5, 5))
                simplex_f = [f0]
                calls = 1

                # Add perturbed points
                for i in range(dim):
                    x_i = x0.copy()
                    x_i[i] += 0.25
                    simplex.append(x_i)
                    f_i = func(np.clip(x_i, -5, 5))
                    simplex_f.append(f_i)
                    calls += 1

                alpha = 1.0   # reflection
                gamma = 2.0   # expansion
                rho = 0.5     # contraction
                sigma = 0.5   # shrink

                while calls < budget_left and time.time() < deadline:
                    # Sort simplex by function value
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]

                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5, 5)
                    f_r = func(x_r)
                    calls += 1

                    if f_r < simplex_f[0]:
                        # Expansion
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        f_e = func(x_e)
                        calls += 1
                        if f_e < f_r:
                            simplex[-1] = x_e
                            simplex_f[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                    elif f_r < simplex_f[-2]:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    else:
                        # Contraction
                        if f_r < simplex_f[-1]:
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            x_c = centroid - rho * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        calls += 1
                        if f_c < min(f_r, simplex_f[-1]):
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex_f[i] = func(np.clip(simplex[i], -5, 5))
                                calls += 1
                                if calls >= budget_left or time.time() > deadline:
                                    break
                            if calls >= budget_left or time.time() > deadline:
                                break

                    # Track best
                    min_idx = np.argmin(simplex_f)
                    if simplex_f[min_idx] < best_f:
                        best_f = simplex_f[min_idx]
                        best_x = simplex[min_idx].copy()

                    if calls >= budget_left:
                        break

                # Return best from simplex
                min_idx = np.argmin(simplex_f)
                return simplex[min_idx], simplex_f[min_idx], calls

            nm_budget = budget - calls_made
            if nm_budget > dim + 1:
                x_nm, f_nm, nm_calls = nm_minimize(func, best_x, nm_budget, deadline)
                calls_made += nm_calls
                if f_nm < best_f:
                    best_f = f_nm
                    best_x = x_nm.copy()

        # Ensure we've made at least one call (should be guaranteed by Phase 1)
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made = 1

        return best_x, best_f
# EVOLVE-BLOCK-END
