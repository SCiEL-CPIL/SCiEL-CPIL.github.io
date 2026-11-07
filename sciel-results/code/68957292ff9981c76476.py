"""Combined optimizer: (1+1)-ES global search + Nelder-Mead local polish for Weierstrass f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB-style Weierstrass f16.
    
    Strategy:
    1. Multi-start to find a promising basin.
    2. (1+1)-ES with adaptive step size for global convergence.
    3. Nelder-Mead simplex refinement for local precision.
    4. Deadline guard to ensure termination.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # --- Phase 0: Multi-Start ---
        candidates = [np.zeros(dim)]
        for _ in range(6):
            candidates.append(rng.uniform(-4.5, 4.5, dim))

        best_x = None
        best_f = np.inf
        for c in candidates:
            fc = func(np.clip(c, -5, 5))
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        x = best_x.copy()
        fx = best_f
        calls = len(candidates)

        # --- Phase 1: (1+1)-ES Global Search (70% of budget) ---
        es_budget = int(budget * 0.7)
        sigma = 1.0
        success_count = 0
        window = 10

        for i in range(es_budget):
            if time.time() >= deadline or calls >= budget:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                success_count += 1

            if (i + 1) % window == 0:
                rate = success_count / window
                if rate > 0.5:
                    sigma *= 1.15
                elif rate < 0.2:
                    sigma *= 0.85
                else:
                    sigma *= 1.0
                success_count = 0

        # --- Phase 2: Nelder-Mead Simplex Refinement ---
        if calls < budget and time.time() < deadline:
            nm_calls = budget - calls
            # Initialize simplex around current best
            step = max(sigma * 0.5, 0.1)
            simplex = [x.copy() for _ in range(dim + 1)]
            simplex_f = [fx]
            for i in range(dim):
                simplex[i + 1] = x.copy()
                simplex[i + 1][i] += step
                simplex_f.append(func(np.clip(simplex[i + 1], -5, 5)))
                calls += 1
                if calls >= budget or time.time() >= deadline:
                    break

            # Nelder-Mead parameters
            alpha = 1.0    # reflection
            gamma = 2.0    # expansion
            rho = 0.5      # contraction
            sigma_nm = 0.5  # shrink

            for _ in range(nm_calls):
                if calls >= budget or time.time() >= deadline:
                    break

                # Sort simplex by function value
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
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
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    if f_c < f_r:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget or time.time() >= deadline:
                                break

                # Update best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < fx:
                    fx = simplex_f[min_idx]
                    x = simplex[min_idx].copy()

        return x, fx
# EVOLVE-BLOCK-END
