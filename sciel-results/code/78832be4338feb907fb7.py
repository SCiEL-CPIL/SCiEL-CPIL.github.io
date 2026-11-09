"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with Nelder-Mead polish for BBOB f20 (Schwefel)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        best_x = rng.uniform(-5, 5, n)
        best_f = func(best_x)
        calls = 1

        # Multi-restart (1+1)-ES
        def es_run(x0, sigma0, max_calls):
            nonlocal calls, best_x, best_f
            x = x0.copy()
            sigma = sigma0
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            no_improve = 0
            while calls < max_calls:
                if time.time() > deadline:
                    break

                # Generate mutant
                z = rng.standard_normal(n)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5, 5)

                fx_trial = func(x_trial)
                calls += 1

                if fx_trial < fx:
                    x = x_trial
                    fx = fx_trial
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    no_improve += 1

                # Adapt step size (1/5+1/2 rule)
                if no_improve < 5:
                    sigma *= 1.1
                elif no_improve > 10:
                    sigma *= 0.9
                sigma = np.clip(sigma, 1e-6, 5.0)

                # Restart if stuck
                if no_improve > 30:
                    break

            return x, fx

        # Nelder-Mead polish
        def nm_polish(x0, max_calls):
            nonlocal calls, best_x, best_f
            if calls >= max_calls:
                return x0, func(x0) if calls < budget else best_f

            # Simple Nelder-Mead
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.1
                simplex.append(xi)

            vals = []
            for i in range(n + 1):
                if calls >= max_calls:
                    break
                v = func(simplex[i])
                vals.append(v)
                calls += 1

            # Only run a few NM iterations to save budget
            for _ in range(10):
                if calls >= max_calls:
                    break
                if len(vals) != n + 1:
                    break

                # Sort simplex
                idx = np.argsort(vals)
                simplex = [simplex[i] for i in idx]
                vals = [vals[i] for i in idx]

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                fw = vals[-1]

                # Reflect
                xr = centroid + 1.0 * (centroid - worst)
                xr = np.clip(xr, -5, 5)
                fxr = func(xr)
                calls += 1

                if fxr < vals[0]:
                    # Expand
                    xe = centroid + 2.0 * (centroid - worst)
                    xe = np.clip(xe, -5, 5)
                    fxe = func(xe)
                    calls += 1
                    if fxe < fxr:
                        simplex[-1] = xe
                        vals[-1] = fxe
                    else:
                        simplex[-1] = xr
                        vals[-1] = fxr
                elif fxr < vals[-2]:
                    simplex[-1] = xr
                    vals[-1] = fxr
                else:
                    # Contract
                    xc = centroid + 0.5 * (worst - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = func(xc)
                    calls += 1
                    if fxc < fw:
                        simplex[-1] = xc
                        vals[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls < max_calls:
                                vals[i] = func(simplex[i])
                                calls += 1

                # Update best
                for i in range(len(simplex)):
                    if vals[i] < best_f:
                        best_f = vals[i]
                        best_x = simplex[i].copy()

            return best_x, best_f

        # Run multiple restarts with different initializations
        n_restarts = 5
        calls_per_restart = (budget - 1) // n_restarts

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Different initialization strategies
            if r == 0:
                x0 = rng.uniform(-4, 4, n)
                sigma0 = 1.0
            elif r == 1:
                x0 = rng.uniform(-3, 3, n)
                sigma0 = 0.5
            elif r == 2:
                x0 = rng.uniform(-5, 5, n)
                sigma0 = 2.0
            elif r == 3:
                x0 = np.zeros(n)
                sigma0 = 1.5
            else:
                x0 = rng.uniform(-4, 4, n)
                sigma0 = 0.8

            max_es_calls = min(calls + calls_per_restart, budget - 20)
            es_run(x0, sigma0, max_es_calls)

            # Polish with NM if we have budget left
            if calls < budget - 10:
                nm_polish(best_x, budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
