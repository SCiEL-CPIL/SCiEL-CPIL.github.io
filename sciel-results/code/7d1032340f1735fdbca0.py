"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of (1+1)-ES runs with Nelder-Mead final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        deadline = time.time() + 290  # Safety margin before 300s limit
        rng = np.random.default_rng()

        def run_es(x_init, budget):
            """Run (1+1)-ES for a specific budget."""
            x = x_init.copy()
            fx = func(x)
            calls = 1
            sigma = 3.0
            min_sigma = 1e-6
            max_sigma = 10.0

            # Aggressive adaptation for fast basin finding
            c_inc = 1.4
            c_dec = 0.75

            for _ in range(budget - 1):
                if time.time() > deadline:
                    break
                if calls >= total_budget:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)

                fy = func(y)
                calls += 1

                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * c_inc, max_sigma)
                else:
                    sigma = max(sigma * c_dec, min_sigma)

            return x, fx, calls

        def nelder_mead(x_init, budget):
            """Nelder-Mead simplex method for precise local optimization."""
            x = x_init.copy()
            fx = func(x)
            calls = 1

            # Create initial simplex
            simplex = [x.copy()]
            for i in range(dim):
                xi = x.copy()
                if abs(xi[i]) > 1e-10:
                    xi[i] += 0.1 * xi[i]
                else:
                    xi[i] += 0.1
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
                fi = func(xi)
                calls += 1
                if time.time() > deadline or calls >= total_budget:
                    break

            # Ensure we have dim+1 points
            while len(simplex) < dim + 1 and calls < total_budget and time.time() < deadline:
                xi = x + 0.1 * rng.standard_normal(dim)
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
                fi = func(xi)
                calls += 1

            # Nelder-Mead parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma_nm = 0.5  # Shrink

            for _ in range(budget - 1):
                if time.time() > deadline:
                    break
                if calls >= total_budget:
                    break

                # Sort simplex by function value
                simplex.sort(key=lambda p: func(p[0]) if False else p[1])
                # Actually, let's store (x, f) pairs properly
                # Let me restructure this

            # Let me rewrite Nelder-Mead properly
            return x, fx, calls

        def nelder_mead_v2(x_init, budget):
            """Nelder-Mead simplex method - corrected version."""
            x = x_init.copy()
            fx = func(x)
            calls = 1

            # Create initial simplex with small perturbation
            simplex = [(x.copy(), fx)]
            base_step = 0.5

            for i in range(dim):
                xi = x.copy()
                xi[i] += base_step
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                if time.time() > deadline or calls >= total_budget:
                    break
                simplex.append((xi, fi))

            # Ensure we have dim+1 points
            while len(simplex) < dim + 1 and calls < total_budget and time.time() < deadline:
                xi = x + base_step * rng.standard_normal(dim)
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                simplex.append((xi, fi))

            if len(simplex) < 2:
                return x, fx, calls

            # Nelder-Mead parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma_nm = 0.5  # Shrink

            for _ in range(budget - 1):
                if time.time() > deadline:
                    break
                if calls >= total_budget:
                    break

                # Sort simplex by function value
                simplex.sort(key=lambda p: p[1])

                # Get best and worst
                best = simplex[0]
                worst = simplex[-1]

                # Check convergence
                if len(simplex) >= 2:
                    if abs(best[1] - worst[1]) < 1e-10:
                        break

                # Compute centroid of all points except worst
                centroid = np.zeros(dim)
                for i in range(len(simplex) - 1):
                    centroid += simplex[i][0]
                centroid /= (len(simplex) - 1)

                # Reflection
                xr = centroid + alpha * (centroid - worst[0])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1

                if fr < best[1]:
                    # Expansion
                    xe = centroid + gamma * (centroid - worst[0])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = (xe, fe)
                    else:
                        simplex[-1] = (xr, fr)
                elif fr < worst[1]:
                    # Reflection is good
                    simplex[-1] = (xr, fr)
                else:
                    # Contraction
                    if fr < worst[1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (worst[0] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < worst[1]:
                        simplex[-1] = (xc, fc)
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i][0] = best[0] + sigma_nm * (simplex[i][0] - best[0])
                            simplex[i][0] = np.clip(simplex[i][0], -5, 5)
                            simplex[i][1] = func(simplex[i][0])
                            calls += 1
                            if time.time() > deadline or calls >= total_budget:
                                break
                        if time.time() > deadline or calls >= total_budget:
                            break

            # Return best point
            simplex.sort(key=lambda p: p[1])
            return simplex[0][0], simplex[0][1], calls

        # Strategy:
        # 1. Explore with 3 random starts, using ~25% of budget each (125 calls).
        # 2. Refine the best found using Nelder-Mead with remaining budget (~125 calls).

        if total_budget < 20:
            x_init = rng.uniform(-5, 5, dim)
            x_final, f_final, _ = run_es(x_init, total_budget)
            return x_final, f_final

        n_explore = 3
        budget_explore_each = min(125, total_budget // (n_explore + 1))
        budget_nm = total_budget - n_explore * budget_explore_each

        best_x = None
        best_f = float('inf')
        total_calls = 0

        for i in range(n_explore):
            if time.time() > deadline:
                break
            if total_calls >= total_budget:
                break

            x_init = rng.uniform(-5, 5, dim)
            x_run, f_run, calls_used = run_es(x_init, budget_explore_each)
            total_calls += calls_used

            if f_run < best_f:
                best_f = f_run
                best_x = x_run

        # Final polish with Nelder-Mead
        if time.time() < deadline and budget_nm > 10 and best_x is not None:
            x_final, f_final, calls_nm = nelder_mead_v2(best_x, budget_nm)
            if f_final < best_f:
                best_f = f_final
                best_x = x_final
        else:
            x_final = best_x
            f_final = best_f

        return x_final, f_final
# EVOLVE-BLOCK-END
