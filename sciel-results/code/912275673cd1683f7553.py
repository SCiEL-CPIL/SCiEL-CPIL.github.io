"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """LHS + Nelder-Mead with adaptive step sizing for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        n = self.dim
        budget = self.budget
        calls = [0]
        best_x = None
        best_f = float('inf')

        def wrapped_func(x):
            nonlocal best_x, best_f
            if calls[0] >= budget or time.time() > deadline:
                return float('inf')
            calls[0] += 1
            x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = func(x_clipped)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Phase 1: Latin Hypercube Sampling for broad exploration
        n_lhs = 40
        if budget > n_lhs:
            rng = np.random.default_rng(42)
            lhs_points = np.zeros((n_lhs, n))
            for d in range(n):
                perm = rng.permutation(n_lhs)
                for i in range(n_lhs):
                    low = -5 + (10 * perm[i] / n_lhs)
                    high = -5 + (10 * (perm[i] + 1) / n_lhs)
                    lhs_points[i, d] = rng.uniform(low, high)

            for x in lhs_points:
                if calls[0] >= budget or time.time() > deadline:
                    break
                wrapped_func(x)

        if best_x is None:
            x0 = np.zeros(n)
            best_f = wrapped_func(x0)
            best_x = x0.copy()

        if calls[0] >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Nelder-Mead with adaptive step sizing
        remaining = budget - calls[0]
        if remaining > 10:
            # Initial simplex size based on problem scale
            simplex_size = 0.5
            
            # Nelder-Mead implementation with adaptive step
            # Start from best_x
            x_current = best_x.copy()
            f_current = best_f
            
            # Create initial simplex
            simplex = [x_current.copy()]
            for d in range(n):
                point = x_current.copy()
                if abs(point[d]) + simplex_size <= 5:
                    point[d] += simplex_size
                else:
                    point[d] -= simplex_size
                simplex.append(point)
            
            # Evaluate simplex
            simplex_f = [f_current]
            for i in range(1, n + 1):
                simplex_f.append(wrapped_func(simplex[i]))
            
            # Nelder-Mead iterations
            max_iter = min(200, remaining // (n + 1))
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            
            for iteration in range(max_iter):
                if time.time() > deadline or calls[0] >= budget:
                    break
                
                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Check convergence
                if abs(simplex_f[-1] - simplex_f[0]) < 1e-12:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect = wrapped_func(x_reflect)
                
                if f_reflect < simplex_f[0]:
                    # Expansion
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand = wrapped_func(x_expand)
                    
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        simplex_f[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        simplex_f[-1] = f_reflect
                elif f_reflect < simplex_f[-2]:
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
                else:
                    # Contraction
                    if f_reflect < simplex_f[-1]:
                        # Outside contraction
                        x_contract = centroid + rho * (x_reflect - centroid)
                    else:
                        # Inside contraction
                        x_contract = centroid - rho * (x_reflect - centroid)
                    
                    x_contract = np.clip(x_contract, -5, 5)
                    f_contract = wrapped_func(x_contract)
                    
                    if f_contract < min(f_reflect, simplex_f[-1]):
                        simplex[-1] = x_contract
                        simplex_f[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = wrapped_func(simplex[i])
                
                # Update best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()
        
        if calls[0] >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Fine-tuning with coordinate descent
        remaining = budget - calls[0]
        if remaining > 5:
            step = 0.05
            for _ in range(min(remaining, 100)):
                if time.time() > deadline or calls[0] >= budget:
                    break
                improved = False
                for d in range(n):
                    for direction in [-step, step]:
                        x_try = best_x.copy()
                        x_try[d] += direction
                        if np.all(x_try >= -5) and np.all(x_try <= 5):
                            val = wrapped_func(x_try)
                            if val < best_f - 1e-15:
                                best_x = x_try
                                best_f = val
                                improved = True
                                break
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-10:
                        break

        # Ensure best_x is within bounds
        best_x = np.clip(best_x, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
