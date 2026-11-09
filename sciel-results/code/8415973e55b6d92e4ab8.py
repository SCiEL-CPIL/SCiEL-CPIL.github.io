"""Multi-start (1+1)-ES with Nelder-Mead polish for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with Nelder-Mead polish for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        n = dim

        best_x = None
        best_f = np.inf
        call_count = 0

        def check_time():
            return time.time() > deadline - 3

        # Phase 1: Diverse initial exploration with structured + random points
        # Use a smaller number of starts to save budget for local search
        n_starts = 12
        start_points = []

        # Center
        start_points.append(np.zeros(dim))

        # Axis-aligned points (±3 in each dimension) - pick a subset
        for i in range(n):
            for val in [-3.0, 3.0]:
                pt = np.zeros(dim)
                pt[i] = val
                start_points.append(pt.copy())

        # Diagonal points
        for val in [-2.0, 2.0]:
            start_points.append(np.full(dim, val))

        # Random points to fill up to n_starts
        while len(start_points) < n_starts:
            start_points.append(rng.uniform(-4, 4, dim))

        # Shuffle for diversity
        idx = rng.permutation(len(start_points))
        start_points = [start_points[i] for i in idx[:n_starts]]

        for sp in start_points:
            if call_count >= budget - 1 or check_time():
                break
            x = np.clip(sp.copy(), -5, 5)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            call_count += 1

        # Phase 2: Adaptive (1+1)-ES from best point
        # Use ~60% of remaining budget for this phase
        remaining = budget - call_count
        es_budget = int(remaining * 0.6)

        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        stagnation = 0
        max_stagnation = 30
        improvement_threshold = 1e-12

        for _ in range(es_budget):
            if call_count >= budget - 1 or check_time():
                break

            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            fy = func(y)
            call_count += 1

            if fy < fx - improvement_threshold:
                x = y
                fx = fy
                sigma *= 1.15
                stagnation = 0
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.9
                stagnation += 1
                if stagnation >= max_stagnation:
                    sigma = max(0.2, min(2.0, sigma * 2.0))
                    stagnation = 0

            # Keep sigma in reasonable bounds
            sigma = max(sigma, 1e-7)
            sigma = min(sigma, 4.0)

        # Phase 3: Nelder-Mead polish
        if call_count < budget - 10 and not check_time():
            # Build initial simplex around best_x
            x0 = best_x.copy()
            f0 = best_f
            simplex = [x0.copy()]
            fs = [f0]

            # Expand simplex along each axis
            step = 0.5
            for i in range(n):
                if call_count >= budget - 5 or check_time():
                    break
                xi = x0.copy()
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                call_count += 1
                simplex.append(xi)
                fs.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            # Nelder-Mead iterations
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_nm = 0.5  # shrink
            max_iters = min(200, budget - call_count - 5)

            for _ in range(max_iters):
                if call_count >= budget - 2 or check_time():
                    break

                # Sort by function value
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]

                # Check convergence
                if fs[0] >= best_f - 1e-10:
                    best_f = fs[0]
                    best_x = simplex[0].copy()
                if np.max(np.abs(np.array(fs) - fs[0])) < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                call_count += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if f_r < fs[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    call_count += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fs[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fs[-1] = f_r
                elif f_r < fs[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fs[-1] = f_r
                else:
                    # Contraction
                    if f_r < fs[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    call_count += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()

                    if f_c < min(f_r, fs[-1]):
                        simplex[-1] = x_c
                        fs[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            if call_count >= budget - 1 or check_time():
                                break
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fs[i] = func(simplex[i])
                            call_count += 1
                            if fs[i] < best_f:
                                best_f = fs[i]
                                best_x = simplex[i].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
