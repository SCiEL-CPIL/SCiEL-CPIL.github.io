"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Hybrid global-local search: random sampling + Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global random sampling to find a good basin
        # Use a subset of budget for global exploration
        n_global = min(50, budget // 4)
        
        best_x = None
        best_f = np.inf
        candidates = []

        for i in range(n_global):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            candidates.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[1])

        # Phase 2: Local refinement using Nelder-Mead on top candidates
        # Use remaining budget for local search
        remaining_budget = budget - n_global

        # Try Nelder-Mead on the best few candidates
        n_local_starts = min(5, len(candidates))
        evals_used = n_global

        for idx in range(n_local_starts):
            if time.time() > deadline:
                break
            if evals_used >= budget:
                break

            x0 = candidates[idx][0].copy()
            
            # Nelder-Mead implementation
            simplex = [x0.copy()]
            f0 = func(x0)
            evals_used += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # Create simplex
            for j in range(dim):
                xi = x0.copy()
                step = 0.5 if x0[j] + 0.5 <= 5 else -0.5
                xi[j] += step
                simplex.append(xi)

            # Nelder-Mead iterations
            nmead_evals = 0
            max_nmead_evals = max(20, remaining_budget // n_local_starts)

            for _ in range(max_nmead_evals):
                if time.time() > deadline:
                    break
                if evals_used >= budget:
                    break

                # Evaluate simplex
                fs = []
                for i_s in range(dim + 1):
                    if i_s == 0:
                        fs.append(f0)
                    else:
                        fx_s = func(simplex[i_s])
                        evals_used += 1
                        nmead_evals += 1
                        fs.append(fx_s)
                        if fx_s < best_f:
                            best_f = fx_s
                            best_x = simplex[i_s].copy()

                # Sort simplex by function value
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                evals_used += 1
                nmead_evals += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if f_r < fs[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    evals_used += 1
                    nmead_evals += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                    else:
                        simplex[-1] = x_r
                elif f_r < fs[-2]:
                    simplex[-1] = x_r
                else:
                    # Contraction
                    rho = 0.5
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    evals_used += 1
                    nmead_evals += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    if f_c < f_r:
                        simplex[-1] = x_c
                    else:
                        # Shrink
                        for i_s in range(1, dim + 1):
                            simplex[i_s] = simplex[0] + 0.5 * (simplex[i_s] - simplex[0])
                            simplex[i_s] = np.clip(simplex[i_s], -5, 5)

        # Phase 3: If budget remains, do fine-grained local search around best
        if evals_used < budget and time.time() < deadline:
            # Use (1+1)-ES with small sigma for final refinement
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            while evals_used < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                evals_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9
                    if sigma < 1e-6:
                        break
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
