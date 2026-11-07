"""Ridge-aware optimizer: Nelder-Mead multi-restart with dynamic coordinate rotation + adaptive polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Nelder-Mead with randomized restarts, dynamic coordinate rotation, and adaptive polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(np.asarray(v, dtype=float), -5.0, 5.0)

        def time_ok():
            return time.time() < deadline

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        max_iter = 200

        def run_nelder_mead(center, initial_step, random_init=False):
            nonlocal best_x, best_f, calls_used

            # Initialize simplex
            simplex = []
            simplex_f = []

            # First point is center
            x0 = clip(center)
            f0 = func(x0)
            calls_used += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            simplex.append(x0)
            simplex_f.append(f0)

            if calls_used >= budget - 5:
                return

            # Other points
            if random_init:
                for i in range(dim):
                    if calls_used >= budget - 5:
                        return
                    direction = rng.normal(0, 1, dim)
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction /= norm
                    x_i = center + initial_step * direction
                    x_i = clip(x_i)
                    f_i = func(x_i)
                    calls_used += 1
                    if f_i < best_f:
                        best_f = f_i
                        best_x = x_i.copy()
                    simplex.append(x_i)
                    simplex_f.append(f_i)
            else:
                for i in range(dim):
                    if calls_used >= budget - 5:
                        return
                    x_i = center.copy()
                    x_i[i] += initial_step
                    x_i = clip(x_i)
                    f_i = func(x_i)
                    calls_used += 1
                    if f_i < best_f:
                        best_f = f_i
                        best_x = x_i.copy()
                    simplex.append(x_i)
                    simplex_f.append(f_i)

            # Main NM loop
            for _ in range(max_iter):
                if not time_ok() or calls_used >= budget - 5:
                    break

                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check for convergence
                if max(simplex_f) - min(simplex_f) < 1e-12:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]

                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = clip(x_r)
                f_r = func(x_r)
                calls_used += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if calls_used >= budget - 5:
                    break

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = clip(x_e)
                    f_e = func(x_e)
                    calls_used += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()

                    if calls_used >= budget - 5:
                        break

                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < worst_f:
                        # Outer contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = clip(x_c)
                        f_c = func(x_c)
                        calls_used += 1
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()

                        if calls_used >= budget - 5:
                            break

                        if f_c < f_r:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                if calls_used >= budget - 5:
                                    return
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = clip(simplex[i])
                                f_s = func(simplex[i])
                                calls_used += 1
                                if f_s < best_f:
                                    best_f = f_s
                                    best_x = simplex[i].copy()
                                simplex_f[i] = f_s
                    else:
                        # Inner contraction
                        x_c = centroid - rho * (centroid - worst)
                        x_c = clip(x_c)
                        f_c = func(x_c)
                        calls_used += 1
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()

                        if calls_used >= budget - 5:
                            break

                        if f_c < worst_f:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                if calls_used >= budget - 5:
                                    return
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = clip(simplex[i])
                                f_s = func(simplex[i])
                                calls_used += 1
                                if f_s < best_f:
                                    best_f = f_s
                                    best_x = simplex[i].copy()
                                simplex_f[i] = f_s

        def polish_phase(center, steps=50):
            """Local random search with decreasing step size."""
            nonlocal best_x, best_f, calls_used

            x = center.copy()
            fx = best_f
            sigma = 0.05
            successes = 0
            window = 0
            WIN = 5

            for _ in range(steps):
                if not time_ok() or calls_used >= budget - 1:
                    break
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = func(y)
                calls_used += 1
                window += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                if window >= WIN:
                    if successes / window > 1.0 / 5.0:
                        sigma *= 1.3
                    else:
                        sigma *= 0.7
                    sigma = float(np.clip(sigma, 1e-8, 1.0))
                    successes = 0
                    window = 0

        # ---------- Phase 1: Multi-start Nelder-Mead with dynamic coordinate rotation ----------
        # First, do a few random starts to find a good region
        n_initial = 10
        for _ in range(n_initial):
            if not time_ok() or calls_used >= budget - 10:
                break
            x_start = rng.uniform(-4.0, 4.0, size=dim)
            run_nelder_mead(x_start, initial_step=0.5, random_init=False)

        # If we have a good best, do a few more targeted restarts
        if best_x is not None and calls_used < budget - 20 and time_ok():
            # Restart around the best point with random initialization
            for _ in range(3):
                if not time_ok() or calls_used >= budget - 15:
                    break
                # Small perturbation around best
                center = clip(best_x + 0.3 * rng.standard_normal(dim))
                run_nelder_mead(center, initial_step=0.3, random_init=True)

        # One more restart with smaller step for fine exploration
        if best_x is not None and calls_used < budget - 10 and time_ok():
            center = best_x.copy()
            run_nelder_mead(center, initial_step=0.1, random_init=True)

        # ---------- Phase 2: Adaptive polish ----------
        if best_x is not None and calls_used < budget and time_ok():
            remaining = budget - calls_used
            steps = min(remaining, 80)
            polish_phase(best_x, steps)

        return best_x, best_f
# EVOLVE-BLOCK-END
