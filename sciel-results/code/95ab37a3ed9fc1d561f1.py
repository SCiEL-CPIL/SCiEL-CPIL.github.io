"""Hybrid Nelder-Mead + adaptive (1+1)-ES optimizer for BBOB f11 Discus."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex followed by adaptive (1+1)-ES for BBOB-style Discus function."""

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

        # Initialize simplex properly (no duplicate)
        simplex = [x0.copy()]
        simplex_f = [f0]
        delta = 0.5
        for i in range(dim):
            xi = x0.copy()
            xi[i] += delta
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

            # Use remaining budget minus reserved for ES
            reserved_for_es = max(100, budget // 5)
            max_nm_iter = max(0, budget - call_count - reserved_for_es)
            
            for _ in range(max_nm_iter):
                if call_count >= budget:
                    break
                if time.time() > deadline - 30:
                    break

                # Sort simplex by function value
                sorted_idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in sorted_idx]
                simplex_f = [simplex_f[i] for i in sorted_idx]

                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = try_eval(xr)
                if fr is None:
                    break

                if fr < simplex_f[0]:
                    # Expansion
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
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Outer contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = try_eval(xc)
                    if fc is None:
                        break
                    if fc < fr:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrinkage
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            fi = try_eval(simplex[i])
                            if fi is None:
                                break
                            simplex_f[i] = fi
                        if call_count >= budget:
                            break

        # Phase 2: Adaptive (1+1)-ES with 1/5th success rule
        if best_x is not None and call_count < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            successes = 0
            total = 0
            window = 10  # Adapt every 10 iterations

            remaining = budget - call_count
            for _ in range(remaining):
                if time.time() > deadline - 10:
                    break
                if call_count >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = try_eval(y)
                if fy is None:
                    break

                total += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1

                # Adapt sigma using 1/5th success rule
                if total % window == 0:
                    success_rate = successes / total
                    if success_rate > 0.5:
                        sigma *= 1.3
                    elif success_rate < 0.2:
                        sigma *= 0.7
                    else:
                        sigma *= 1.0
                    # Clamp sigma
                    sigma = max(1e-6, min(sigma, 5.0))
                    successes = 0
                    total = 0

        # Phase 3: Fine-grained polish with very small steps
        if best_x is not None and call_count < budget:
            x = best_x.copy()
            fx = best_f
            sigma = max(1e-4, 0.01)
            polish_budget = min(50, budget - call_count)
            for _ in range(polish_budget):
                if time.time() > deadline - 2:
                    break
                if call_count >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.9

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
