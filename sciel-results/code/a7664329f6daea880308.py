"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size + local Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 310.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Start at center
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1
        sigma = 2.0

        # Phase 1: (1+1)-ES with 1/5 success rule - fast exploration
        # Reserve ~20% of budget for local polish
        es_budget = int(budget * 0.8)
        
        while calls < es_budget:
            if time.time() > deadline - 5:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3  # increase step on success
            else:
                sigma *= 0.82  # decrease step on failure
            
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()

        # Phase 2: Local Nelder-Mead polish around best solution
        if calls < budget - 5 and time.time() < deadline - 5:
            x_nm = best_x.copy()
            
            # Initialize simplex
            simplex = [x_nm.copy()]
            for i in range(dim):
                pert = x_nm.copy()
                step = 0.5 if abs(x_nm[i]) < 4.5 else 0.1  # smaller step near boundaries
                pert[i] += step
                simplex.append(np.clip(pert, -5, 5))

            fvals = [func(s) for s in simplex]
            calls += dim + 1

            # Nelder-Mead iterations
            max_nm_iters = min(30, budget - calls)
            for _ in range(max_nm_iters):
                if time.time() > deadline - 5 or calls >= budget:
                    break
                
                # Sort simplex
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = np.clip(2 * centroid - simplex[-1], -5, 5)
                fr = func(xr)
                calls += 1
                if fr < fvals[0]:
                    # Expansion
                    xe = np.clip(centroid + 2 * (xr - centroid), -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = np.clip(centroid + 0.5 * (simplex[-1] - centroid), -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = np.clip(0.5 * (simplex[i] + simplex[0]), -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1

                if fvals[0] < best_fx:
                    best_fx = fvals[0]
                    best_x = simplex[0].copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
