"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio optimizer: multi-start local search + adaptive evolution strategy with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        # Counters
        calls = [0]

        def safe_func(x):
            if calls[0] >= budget:
                raise StopIteration
            calls[0] += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        # ---- Phase 1: Multi-start with diverse initializations ----
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # Generate diverse starts: grid points, random, and structured patterns
        starts = []
        # Origin and axis-aligned points
        starts.append(np.array([0.0] * dim))
        starts.append(np.array([2.0] * dim))
        starts.append(np.array([-2.0] * dim))
        starts.append(np.array([3.0] * dim))
        starts.append(np.array([-3.0] * dim))
        # Random starts
        for _ in range(8):
            starts.append(rng.uniform(-4, 4, dim))
        # Structured: alternating signs
        alt = np.array([2.0 if i % 2 == 0 else -2.0 for i in range(dim)])
        starts.append(alt)
        starts.append(-alt)

        for s in starts:
            if calls[0] >= budget:
                break
            x = np.clip(s, -5, 5)
            try:
                fx = safe_func(x)
            except StopIteration:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # ---- Phase 2: Coordinate descent with adaptive step sizes per dimension ----
        steps = np.ones(dim)
        max_cd_iters = 30
        for it in range(max_cd_iters):
            if calls[0] >= budget or time.time() > deadline:
                break
            improved = False
            for d in range(dim):
                if calls[0] >= budget or time.time() > deadline:
                    break
                # Try both directions with current step
                for sign in (1, -1):
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    x2 = best_x.copy()
                    x2[d] = np.clip(best_x[d] + sign * steps[d], -5, 5)
                    try:
                        f2 = safe_func(x2)
                    except StopIteration:
                        break
                    if f2 < best_f:
                        best_f = f2
                        best_x = x2
                        improved = True
                        # Reduce step size when improving (zooming in)
                        steps[d] *= 0.7
                # If no improvement in either direction, try smaller step
                if not improved:
                    steps[d] *= 0.5

            # If no dimension improved at all, shrink all steps
            if not improved:
                steps *= 0.8
                if np.all(steps < 1e-5):
                    break

        # ---- Phase 3: Adaptive evolution strategy with restarts ----
        center = best_x.copy()
        cf = best_f
        sigma = 1.0
        lam = 6  # population size
        consecutive_failures = 0
        max_failures = 10

        while calls[0] < budget and time.time() < deadline:
            # Generate population
            improved_any = False
            best_child_x = center.copy()
            best_child_f = cf

            for i in range(lam):
                if calls[0] >= budget or time.time() > deadline:
                    break
                # Mix of local and global perturbations
                if i < lam // 2:
                    # Local perturbation
                    child = np.clip(center + sigma * rng.standard_normal(dim), -5, 5)
                else:
                    # Larger perturbation for exploration
                    child = np.clip(center + sigma * 2.0 * rng.standard_normal(dim), -5, 5)
                try:
                    cf_c = safe_func(child)
                except StopIteration:
                    break
                if cf_c < best_child_f:
                    best_child_f = cf_c
                    best_child_x = child.copy()

            if best_child_f < cf:
                cf = best_child_f
                center = best_child_x
                best_f = cf
                best_x = center.copy()
                improved_any = True
                consecutive_failures = 0
                # Slightly increase sigma on success
                sigma = min(sigma * 1.1, 5.0)
            else:
                consecutive_failures += 1
                # Decrease sigma on failure
                sigma *= 0.85

            # Restart if stuck too long
            if consecutive_failures >= max_failures:
                if calls[0] >= budget or time.time() > deadline:
                    break
                # Perturb center significantly
                center = np.clip(best_x + 1.5 * rng.standard_normal(dim), -5, 5)
                try:
                    cf_c = safe_func(center)
                except StopIteration:
                    break
                if cf_c < cf:
                    cf = cf_c
                    best_f = cf
                    best_x = center.copy()
                sigma = 1.0
                consecutive_failures = 0

            if sigma < 1e-6:
                break

        # ---- Phase 4: Final high-precision polish with Nelder-Mead ----
        if calls[0] < budget - 10 and time.time() < deadline - 10:
            # Implement a simple Nelder-Mead for final polish
            simplex = [best_x.copy()]
            for d in range(dim):
                pt = best_x.copy()
                pt[d] += 0.1
                pt = np.clip(pt, -5, 5)
                simplex.append(pt)

            # Evaluate simplex
            simplex_f = []
            for pt in simplex:
                if calls[0] >= budget or time.time() > deadline:
                    break
                try:
                    simplex_f.append(safe_func(pt))
                except StopIteration:
                    break

            if len(simplex_f) == dim + 1:
                # Nelder-Mead iterations
                alpha, gamma, rho = 0.5, 2.0, -0.5
                for nm_iter in range(50):
                    if calls[0] >= budget or time.time() > deadline:
                        break

                    # Sort simplex
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]

                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                    try:
                        f_r = safe_func(x_r)
                    except StopIteration:
                        break

                    if simplex_f[0] <= f_r < simplex_f[-2]:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    elif f_r < simplex_f[0]:
                        # Expansion
                        x_e = np.clip(centroid + gamma * (x_r - centroid), -5, 5)
                        try:
                            f_e = safe_func(x_e)
                        except StopIteration:
                            break
                        if f_e < f_r:
                            simplex[-1] = x_e
                            simplex_f[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                    else:
                        # Contraction
                        x_c = np.clip(centroid + rho * (simplex[-1] - centroid), -5, 5)
                        try:
                            f_c = safe_func(x_c)
                        except StopIteration:
                            break
                        if f_c < simplex_f[-1]:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5, 5)
                                try:
                                    simplex_f[i] = safe_func(simplex[i])
                                except StopIteration:
                                    break

                    # Update best
                    min_idx = np.argmin(simplex_f)
                    if simplex_f[min_idx] < best_f:
                        best_f = simplex_f[min_idx]
                        best_x = simplex[min_idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
