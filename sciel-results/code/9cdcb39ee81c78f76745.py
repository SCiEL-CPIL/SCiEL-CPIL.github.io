"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid (1+1)-ES with Nelder-Mead exploitation and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, budget

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global exploration with (1+1)-ES and 1/5 success rule
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls = 1

        successes = 0
        window = 10
        sigma = 2.0
        sigma_min = 1e-6
        sigma_max = 5.0

        # Run (1+1)-ES until we've used ~35% of budget or time runs low
        es_budget = int(budget * 0.35)
        while calls < es_budget:
            if time.time() >= deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
                sigma = min(sigma * 1.05, sigma_max)
            else:
                sigma = max(sigma * 0.95, sigma_min)

            if calls % window == 0:
                if successes >= window / 5.0:
                    sigma = min(sigma * 1.2, sigma_max)
                else:
                    sigma = max(sigma * 0.8, sigma_min)
                successes = 0

        # Phase 2: Nelder-Mead simplex method for local exploitation
        if calls < budget and time.time() < deadline - 10:
            # Initialize simplex around current best point
            alpha = 0.5  # Expansion factor
            gamma = 2.0  # Expansion factor
            rho = -0.5   # Reflection factor
            sigma_nm = max(sigma * 0.3, 0.1)  # Initial simplex size
            
            # Create initial simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x
            for i in range(1, dim + 1):
                simplex[i] = x.copy()
                simplex[i][i - 1] += sigma_nm
            
            # Evaluate simplex
            simplex_f = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls >= budget or time.time() >= deadline:
                    break
                simplex_f[i] = func(simplex[i])
                calls += 1
            
            # Track best found in NM
            nm_best_x = x.copy()
            nm_best_f = fx
            
            # Nelder-Mead iterations
            nm_max_iter = int((budget - calls) * 0.4)
            nm_iter = 0
            
            while nm_iter < nm_max_iter and calls < budget and time.time() < deadline - 5:
                nm_iter += 1
                
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                
                # Update best
                if simplex_f[0] < nm_best_f:
                    nm_best_f = simplex_f[0]
                    nm_best_x = simplex[0].copy()
                
                # Check convergence
                if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + rho * (simplex[-1] - centroid)
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1
                
                if calls >= budget or time.time() >= deadline - 5:
                    break
                
                # If reflection is better than second best, accept
                if simplex_f[1] <= fr < simplex_f[0]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                # If reflection is best, try expansion
                elif fr < simplex_f[0]:
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    
                    if calls >= budget or time.time() >= deadline - 5:
                        break
                    
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                # If reflection is worst, try contraction
                else:
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    
                    if calls >= budget or time.time() >= deadline - 5:
                        break
                    
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink towards best point
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls < budget and time.time() < deadline - 5:
                                simplex_f[i] = func(simplex[i])
                                calls += 1
                        if calls >= budget or time.time() >= deadline - 5:
                            break
                
                # Update overall best
                if simplex_f[0] < nm_best_f:
                    nm_best_f = simplex_f[0]
                    nm_best_x = simplex[0].copy()
                
                # If we found something better than ES result, use it
                if nm_best_f < fx:
                    x, fx = nm_best_x.copy(), nm_best_f

        # Phase 3: Final polish with very small random steps
        if calls < budget and time.time() < deadline - 2:
            sigma_polish = max(1e-4, 1e-6)
            while calls < budget:
                if time.time() >= deadline:
                    break
                y = np.clip(x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_polish = min(sigma_polish * 1.05, 1e-3)
                else:
                    sigma_polish = max(sigma_polish * 0.95, 1e-10)

        return x, fx
# EVOLVE-BLOCK-END
