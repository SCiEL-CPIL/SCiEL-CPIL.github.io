"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-start Nelder-Mead exploiting Rastrigin's periodic structure."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def try_point(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        def time_ok():
            return time.time() < deadline

        # --- Phase 1: Structured sampling exploiting Rastrigin's periodicity ---
        # Rastrigin: f(x) = A*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # The cos term has period 2, so minima occur near integer values (shifted by hidden offset).
        # We sample a grid that covers the [-4,4] range with spacing ~1.0 to hit near all basins.
        
        # Generate candidate starting points:
        # 1) Center of domain
        # 2) Grid points at integer and half-integer positions in [-4, 4]
        # 3) A few random-like points using deterministic Halton sequence

        candidates = []
        
        # Center
        candidates.append(np.zeros(dim))
        
        # Grid: for each dimension, try -4, -3, -2, -1, 0, 1, 2, 3, 4
        # That's 9^5 = 59049 points - too many. Use a subset.
        # Instead, use a stratified approach: for each dimension, pick 3 levels
        # and generate a subset of combinations.
        
        # Better: Use a space-filling sequence (Halton) to get diverse points,
        # but also include structured points at integer coordinates.
        
        # Halton sequence for deterministic space-filling
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Generate 50 Halton points in [0,1]^dim, map to [-4,4]^dim
        n_halton = 50
        for i in range(1, n_halton + 1):
            point = np.array([halton(i, base) for base in [2, 3, 5, 7, 11][:dim]])
            point = -4.0 + 8.0 * point  # map to [-4, 4]
            candidates.append(point)

        # Add structured integer grid points (subset)
        # For each dimension, try values -4, -2, 0, 2, 4
        # Generate a subset of the full 5^5 grid using a stride
        grid_levels = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
        # Instead of all 3125, pick every 7th combination
        count = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    for l in range(5):
                        for m in range(5):
                            if (i + 2*j + 3*k + 4*l + 5*m) % 7 == 0:
                                candidates.append(np.array([grid_levels[i], grid_levels[j], 
                                                          grid_levels[k], grid_levels[l], grid_levels[m]]))
                            count += 1

        # Evaluate all candidates, keeping track of best
        for point in candidates:
            if not time_ok() or calls_made >= budget:
                break
            try_point(point)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made = 1

        # --- Phase 2: Multi-start Nelder-Mead from top K diverse starting points ---
        # We'll run Nelder-Mead from several of the best starting points found in Phase 1,
        # but to save budget, we'll do a quick coordinate-wise refinement first,
        # then launch Nelder-Mead from the best few.

        # First, do a quick coordinate-wise refinement on the global best
        if time_ok() and calls_made < budget:
            for _pass in range(2):
                if not time_ok() or calls_made >= budget:
                    break
                for d in range(dim):
                    if not time_ok() or calls_made >= budget:
                        break
                    local_best_x = best_x.copy()
                    local_best_f = best_f
                    # Try offsets
                    for offset in [-1.0, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1.0]:
                        if not time_ok() or calls_made >= budget:
                            break
                        x_try = local_best_x.copy()
                        x_try[d] = local_best_x[d] + offset
                        x_try = np.clip(x_try, -5.0, 5.0)
                        fx = func(x_try)
                        calls_made += 1
                        if fx < local_best_f:
                            local_best_f = fx
                            local_best_x = x_try.copy()
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()

        # --- Phase 3: Nelder-Mead polish ---
        if best_x is not None and time_ok() and calls_made < budget:
            def nm_minimize(func, x0, budget_left, deadline):
                dim = len(x0)
                simplex = [x0.copy()]
                f0 = func(np.clip(x0, -5, 5))
                simplex_f = [f0]
                calls = 1

                # Initialize simplex with small perturbations
                for i in range(dim):
                    x_i = x0.copy()
                    x_i[i] += 0.1
                    simplex.append(x_i)
                    f_i = func(np.clip(x_i, -5, 5))
                    simplex_f.append(f_i)
                    calls += 1

                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5

                while calls < budget_left and time.time() < deadline:
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]

                    centroid = np.mean(simplex[:-1], axis=0)

                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5, 5)
                    f_r = func(x_r)
                    calls += 1

                    if f_r < simplex_f[0]:
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

                    min_idx = np.argmin(simplex_f)
                    if simplex_f[min_idx] < best_f:
                        best_f = simplex_f[min_idx]
                        best_x = simplex[min_idx].copy()

                    if calls >= budget_left:
                        break

                min_idx = np.argmin(simplex_f)
                return simplex[min_idx], simplex_f[min_idx], calls

            nm_budget = budget - calls_made
            if nm_budget > dim + 1:
                x_nm, f_nm, nm_calls = nm_minimize(func, best_x, nm_budget, deadline)
                calls_made += nm_calls
                if f_nm < best_f:
                    best_f = f_nm
                    best_x = x_nm.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made = 1

        return best_x, best_f
# EVOLVE-BLOCK-END
