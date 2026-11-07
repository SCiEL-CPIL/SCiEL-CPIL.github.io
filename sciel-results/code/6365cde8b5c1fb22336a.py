"""Black-box continuous optimizer on BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Nelder-Mead + (1+1)-ES optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        # Helper to evaluate with budget tracking
        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial exploration with 30 random points
        n_global = min(30, budget)
        rng = np.random.default_rng(42)
        candidates = []
        for i in range(n_global):
            x = rng.uniform(-5, 5, dim)
            fx = eval_func(x)
            candidates.append((fx, x))
            if time.time() > deadline:
                break

        # Sort by function value and take top 3 as starting points
        candidates.sort(key=lambda t: t[0])
        top_k = min(3, len(candidates))
        starts = [candidates[i][1].copy() for i in range(top_k)]

        # Phase 2: Run Nelder-Mead from each of the top 3 starting points
        # Allocate remaining budget evenly
        remaining = budget - calls
        budget_per_start = max(1, remaining // top_k)

        for si in range(top_k):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Initialize simplex around the starting point
            x0 = starts[si]
            step = 0.5
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += step
                simplex.append(xi)
            # One extra point with a bigger step in a diagonal direction
            xd = x0.copy()
            xd += 0.3 * np.ones(dim)
            simplex.append(xd)

            # Evaluate all simplex points
            fx_list = []
            for i in range(len(simplex)):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break
                xi = simplex[i].copy()
                fi = eval_func(xi)
                fx_list.append(fi)
                if fi == np.inf:
                    break

            if len(fx_list) < 2:
                continue

            n_simplex = len(simplex)

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            # Nelder-Mead iterations
            max_iters = budget_per_start - len(fx_list)
            for iteration in range(max_iters):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break

                # Sort simplex by function value
                order = np.argsort(fx_list)
                simplex = [simplex[i] for i in order]
                fx_list = [fx_list[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_fx = fx_list[-1]
                best_fx_simplex = fx_list[0]

                # Reflection
                xr = centroid + alpha * (centroid - worst)
                xr = np.clip(xr, -5, 5)
                fr = eval_func(xr)
                if calls >= budget:
                    break

                if fr < best_fx_simplex:
                    # Try expansion
                    xe = centroid + gamma * (centroid - worst)
                    xe = np.clip(xe, -5, 5)
                    fe = eval_func(xe)
                    if calls >= budget:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        fx_list[-1] = fe
                    else:
                        simplex[-1] = xr
                        fx_list[-1] = fr
                elif fr < worst_fx:
                    simplex[-1] = xr
                    fx_list[-1] = fr
                else:
                    # Contraction
                    if fr < worst_fx:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = eval_func(xc)
                        if calls >= budget:
                            break
                        if fc <= fr:
                            simplex[-1] = xc
                            fx_list[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n_simplex):
                                if time.time() > deadline:
                                    break
                                if calls >= budget:
                                    break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fi = eval_func(simplex[i])
                                fx_list[i] = fi
                    else:
                        # Inside contraction
                        xc = centroid - rho * (worst - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = eval_func(xc)
                        if calls >= budget:
                            break
                        if fc < worst_fx:
                            simplex[-1] = xc
                            fx_list[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n_simplex):
                                if time.time() > deadline:
                                    break
                                if calls >= budget:
                                    break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fi = eval_func(simplex[i])
                                fx_list[i] = fi

                # Check for convergence: simplex is very small
                spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                if np.all(spread < 1e-8):
                    break

        # Phase 3: Final refinement with (1+1)-ES from the best point found
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            success_count = 0
            attempts = 0

            for step in range(budget - calls):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)
                if fy == np.inf:
                    break

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    sigma *= 1.25
                else:
                    sigma *= 0.85

                # Adaptive sigma adjustment every 10 steps
                attempts += 1
                if attempts % 10 == 0:
                    rate = success_count / attempts
                    if rate > 0.5:
                        sigma *= 1.1
                    elif rate < 0.1:
                        sigma *= 0.9
                    success_count = 0
                    attempts = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
