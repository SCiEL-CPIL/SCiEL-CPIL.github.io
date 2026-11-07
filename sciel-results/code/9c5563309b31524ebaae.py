"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer for Rosenbrock valley."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def nelder_mead(x0, max_calls, initial_alpha=1.0):
            """Nelder-Mead with adaptive restart and step size."""
            nonlocal calls, best_x, best_f

            if calls >= budget or time.time() > deadline:
                return x0, np.inf

            # Initial simplex
            alpha = initial_alpha
            simplex = []
            simplex_f = []

            # Generate initial points
            for i in range(n + 1):
                if calls >= budget or time.time() > deadline:
                    break
                if i == 0:
                    x = x0.copy()
                else:
                    x = x0.copy()
                    x[i - 1] += alpha
                x = np.clip(x, -5, 5)
                f = func(x)
                calls += 1
                simplex.append(x)
                simplex_f.append(f)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            if len(simplex) < 2:
                return x0, np.inf

            # Sort simplex by fitness
            idx = np.argsort(simplex_f)
            simplex = [simplex[i] for i in idx]
            simplex_f = [simplex_f[i] for i in idx]

            # Nelder-Mead parameters
            alpha_nm = 1.0  # Reflection
            gamma = 2.0     # Expansion
            rho = 0.5       # Contraction
            sigma_nm = 0.5  # Shrink

            stagnation_count = 0
            max_stagnation = 10

            while calls < budget and time.time() < deadline:
                # Check convergence
                f_min = simplex_f[0]
                f_max = simplex_f[-1]
                if f_max - f_min < 1e-12 * max(1.0, abs(f_min)):
                    break

                # 1. Reflection
                centroid = np.mean(simplex[1:], axis=0)
                x_r = centroid + alpha_nm * (centroid - simplex[0])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if f_r < simplex_f[-1] and f_r >= simplex_f[0]:
                    # Accept reflection
                    simplex[0] = x_r
                    simplex_f[0] = f_r
                    stagnation_count = 0
                elif f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    
                    if f_e < f_r:
                        simplex[0] = x_e
                        simplex_f[0] = f_e
                    else:
                        simplex[0] = x_r
                        simplex_f[0] = f_r
                    stagnation_count = 0
                else:
                    # Contraction
                    # Outside contraction
                    if f_r < simplex_f[-1]:
                        x_c = centroid + 0.5 * (x_r - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        calls += 1
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()
                        
                        if f_c < f_r:
                            simplex[0] = x_c
                            simplex_f[0] = f_c
                            stagnation_count = 0
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex_f[i] = func(simplex[i])
                                calls += 1
                                if simplex_f[i] < best_f:
                                    best_f = simplex_f[i]
                                    best_x = simplex[i].copy()
                            stagnation_count += 1
                    else:
                        # Inside contraction
                        x_c = centroid - 0.5 * (centroid - simplex[0])
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        calls += 1
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()

                        if f_c < simplex_f[0]:
                            simplex[0] = x_c
                            simplex_f[0] = f_c
                            stagnation_count = 0
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex_f[i] = func(simplex[i])
                                calls += 1
                                if simplex_f[i] < best_f:
                                    best_f = simplex_f[i]
                                    best_x = simplex[i].copy()
                            stagnation_count += 1

                # Sort simplex again
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]

                if stagnation_count > max_stagnation:
                    break

            return simplex[0], simplex_f[0]

        # Multi-start strategy
        # 1. Start at origin
        x_start = np.zeros(n)
        f_start = func(x_start)
        calls += 1
        best_x = x_start.copy()
        best_f = f_start

        if calls < budget and time.time() < deadline:
            nelder_mead(x_start, budget - calls, initial_alpha=1.0)

        # 2. Restart with diverse points if not converged
        # The minimizer is in [-4, 4]^5
        num_restarts = 5
        for i in range(num_restarts):
            if calls >= budget or time.time() > deadline:
                break
            
            if best_f < 1e-6:
                break

            # Generate a new start point
            # Mix of random and perturbed best
            if i % 2 == 0:
                x_new = rng.uniform(-4, 4, n)
            else:
                x_new = best_x + rng.normal(0, 2.0, n)
                x_new = np.clip(x_new, -5, 5)
            
            f_new = func(x_new)
            calls += 1
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

            if calls < budget and time.time() < deadline:
                # Use smaller alpha for refinement if we have a good best
                alpha = 0.5 if best_f < 10 else 1.0
                nelder_mead(x_new, budget - calls, initial_alpha=alpha)

        return best_x, best_f
# EVOLVE-BLOCK-END
