"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with aggressive sigma adaptation and adaptive Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_evals = 0

        def check_time():
            return time.time() > deadline

        # Phase 1: Quick random search to find a good starting point
        n_random = min(30, budget // 8)
        for i in range(n_random):
            if check_time() or total_evals >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            total_evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: (1+1)-ES with 1/5 success rule
        # For f14 (different powers), the function is unimodal but anisotropic
        # Start with a moderate sigma and let it adapt
        sigma = 1.5
        c_sigma = 1.0 / (dim + 3)  # adaptation rate
        d_sigma = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 1)) - 1) + c_sigma
        successes = 0
        window = 10  # window for success counting

        # Run (1+1)-ES until we run low on budget or stagnate
        es_evals = 0
        max_es_evals = budget - total_evals - 50  # reserve some for NM
        stagnation_counter = 0
        stagnation_limit = 50  # restart/stagnation threshold

        while total_evals < budget - 50 and not check_time():
            # Generate a single offspring
            step = sigma * rng.standard_normal(dim)
            y = best_x + step
            y = np.clip(y, -5, 5)
            f_y = func(y)
            total_evals += 1
            es_evals += 1

            if f_y < best_f:
                successes += 1
                best_f = f_y
                best_x = y.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Adapt sigma based on success rate over last 'window' evaluations
            if es_evals % window == 0:
                success_rate = successes / window
                # 1/5 success rule: if success rate > 1/5, increase sigma
                # if success rate < 1/5, decrease sigma
                if success_rate > 0.2:
                    sigma *= 1.15
                elif success_rate < 0.2:
                    sigma *= 0.85
                sigma = np.clip(sigma, 1e-6, 10.0)
                successes = 0

            # If stagnating too long, try a larger perturbation
            if stagnation_counter > stagnation_limit:
                # Try a random jump
                jump = 2.0 * rng.standard_normal(dim)
                y2 = best_x + jump
                y2 = np.clip(y2, -5, 5)
                f_y2 = func(y2)
                total_evals += 1
                if f_y2 < best_f:
                    best_f = f_y2
                    best_x = y2.copy()
                stagnation_counter = 0
                sigma = max(sigma, 1.0)

        # Phase 3: Nelder-Mead for local refinement
        if total_evals < budget - 10 and not check_time():
            # Build initial simplex around best_x
            simplex = [best_x.copy()]
            for i in range(dim):
                point = best_x.copy()
                point[i] += 0.1 if best_x[i] < 4.9 else -0.1
                simplex.append(point)
            simplex = np.array(simplex)
            f_simplex = np.array([func(x) for x in simplex])
            total_evals += dim + 1

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_nm = 0.5  # shrink

            nm_iter = 0
            max_nm_iter = 200
            while total_evals < budget - 5 and not check_time() and nm_iter < max_nm_iter:
                nm_iter += 1
                # Sort simplex by function value
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]

                if f_simplex[0] < best_f:
                    best_f = f_simplex[0]
                    best_x = simplex[0].copy()

                # Centroid of all points except worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                worst = simplex[-1]
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                total_evals += 1

                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    total_evals += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_simplex[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    total_evals += 1
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            total_evals += 1
                            if total_evals >= budget:
                                break

            # Update best from simplex
            order = np.argsort(f_simplex)
            if f_simplex[order[0]] < best_f:
                best_f = f_simplex[order[0]]
                best_x = simplex[order[0]].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
