"""Hybrid Nelder-Mead + BFGS polish + (1+1)-ES optimizer for BBOB f11 Discus."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead simplex, BFGS polish, then (1+1)-ES for BBOB-style Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Nelder-Mead Simplex Method
        x0 = np.zeros(dim)
        f0 = try_eval(x0)
        if f0 is None:
            return best_x, best_f

        simplex = [x0.copy()]
        simplex_f = [f0]

        for i in range(dim):
            xi = x0.copy()
            xi[i] += 0.5
            fi = try_eval(xi)
            if fi is None:
                break
            simplex.append(xi)
            simplex_f.append(fi)

        if len(simplex) >= dim + 1:
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            nm_budget = max(50, budget - call_count - 80)
            no_improve = 0
            for _ in range(nm_budget):
                if call_count >= budget:
                    break
                if time.time() > deadline - 30:
                    break

                sorted_idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in sorted_idx]
                simplex_f = [simplex_f[i] for i in sorted_idx]

                centroid = np.mean(simplex[:-1], axis=0)

                xr = centroid + alpha * (centroid - simplex[-1])
                fr = try_eval(xr)
                if fr is None:
                    break

                improved = False
                if fr < simplex_f[0]:
                    if fr < simplex_f[-2]:
                        xe = centroid + gamma * (centroid - simplex[-1])
                        fe = try_eval(xe)
                        if fe is None:
                            break
                        if fe < fr:
                            simplex[-1] = xe
                            simplex_f[-1] = fe
                        else:
                            simplex[-1] = xr
                            simplex_f[-1] = fr
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    improved = True
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                    improved = True
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = try_eval(xc)
                    if fc is None:
                        break
                    if fc < fr:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                        improved = True
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            fi = try_eval(simplex[i])
                            if fi is None:
                                break
                            simplex_f[i] = fi
                        if call_count >= budget:
                            break

                if improved:
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= 10:
                    break

        # Phase 2: BFGS polish using scipy.optimize.minimize
        if best_x is not None and call_count < budget - 20:
            def wrapped_func(x):
                if call_count >= budget:
                    return best_f
                return try_eval(x)

            try:
                res = minimize(wrapped_func, best_x.copy(), method='BFGS',
                               options={'maxiter': 50, 'gtol': 1e-6})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: (1+1)-ES polish with adaptive sigma
        if best_x is not None and call_count < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(budget - call_count):
                if time.time() > deadline - 5:
                    break
                if call_count >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.85

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
