"""Ridge-aware optimizer using Nelder-Mead with randomized initialization and adaptive polishing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead optimizer with strict budget control and adaptive refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        dim = self.dim
        budget = self.budget

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
        sigma = 0.5
        max_iter = 150

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
                # Generate random orthogonal-ish directions
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
                # Axis-aligned
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
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
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
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = clip(simplex[i])
                                f_s = func(simplex[i])
                                calls_used += 1
                                if f_s < best_f:
                                    best_f = f_s
                                    best_x = simplex[i].copy()
                                simplex_f[i] = f_s

        def adaptive_polish(center, steps=80):
            """Adaptive random search with momentum and step size adjustment."""
            nonlocal best_x, best_f, calls_used

            x = clip(center).copy()
            f_curr = func(x)
            calls_used += 1
            if f_curr < best_f:
                best_f = f_curr
                best_x = x.copy()

            # Adaptive step size with momentum
            sigma = 0.5
            sigma_min = 1e-4
            sigma_max = 2.0
            velocity = np.zeros(dim)
            momentum = 0.6

            no_improve_count = 0
            success_count = 0
            total_count = 0
            window_size = 10

            for _ in range(steps):
                if not time_ok() or calls_used >= budget - 2:
                    break

                # Generate candidate with adaptive sigma and momentum
                noise = rng.standard_normal(dim)
                if np.linalg.norm(velocity) > 1e-8:
                    dir_unit = velocity / np.linalg.norm(velocity)
                    noise = 0.7 * noise + 0.3 * dir_unit * np.linalg.norm(noise)

                candidate = x + sigma * noise
                candidate = clip(candidate)
                f_new = func(candidate)
                calls_used += 1

                if f_new < f_curr:
                    # Improvement
                    velocity = momentum * velocity + (1 - momentum) * (candidate - x)
                    x = candidate
                    f_curr = f_new
                    sigma = min(sigma * 1.15, sigma_max)
                    success_count += 1

                    if f_new < best_f:
                        best_f = f_new
                        best_x = x.copy()
                    no_improve_count = 0
                else:
                    # No improvement
                    velocity = momentum * velocity
                    sigma = max(sigma * 0.92, sigma_min)
                    no_improve_count += 1

                total_count += 1

                # Periodically adjust sigma based on success rate
                if total_count % window_size == 0:
                    success_rate = success_count / total_count
                    if success_rate > 0.7:
                        sigma = min(sigma * 1.1, sigma_max)
                    elif success_rate < 0.3:
                        sigma = max(sigma * 0.85, sigma_min)
                    success_count = 0
                    total_count = 0

                # Occasionally try a larger jump to escape local optima
                if _ % 30 == 29 and sigma < 1.5:
                    big_noise = rng.standard_normal(dim) * 1.5
                    big_candidate = clip(x + big_noise)
                    if calls_used < budget - 2:
                        big_fy = func(big_candidate)
                        calls_used += 1
                        if big_fy < f_curr:
                            x = big_candidate
                            f_curr = big_fy
                            sigma = min(sigma * 1.3, sigma_max)
                            if big_fy < best_f:
                                best_f = big_fy
                                best_x = x.copy()

        # Phase 1: Coarse search with randomized initialization
        center = np.zeros(dim)
        run_nelder_mead(center, 3.0, random_init=True)

        # Phase 2: Finer search / Restart from best
        if calls_used < budget - 50 and best_x is not None:
            # Restart from best with smaller step
            run_nelder_mead(best_x, 1.0, random_init=False)

        # Phase 3: Another restart with random init for exploration
        if calls_used < budget - 40 and best_x is not None:
            perturbed_center = best_x + rng.normal(0, 0.5, dim)
            run_nelder_mead(perturbed_center, 0.5, random_init=True)

        # Phase 4: Adaptive polish with momentum
        if calls_used < budget - 10 and best_x is not None:
            adaptive_polish(best_x, steps=80)

        # Phase 5: Final fine polish if budget allows
        if calls_used < budget - 5 and best_x is not None:
            # Very fine local search
            fine_center = best_x.copy()
            for _ in range(20):
                if not time_ok() or calls_used >= budget - 2:
                    break
                step = 0.01 * rng.uniform(0.5, 2.0)
                direction = rng.normal(0, 1, dim)
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
                x_new = clip(fine_center + step * direction)
                f_new = func(x_new)
                calls_used += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    fine_center = x_new.copy()

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
