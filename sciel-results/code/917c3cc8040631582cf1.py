"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Nelder-Mead optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 325.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0

        # Track best solution
        best_x = np.full(dim, 5.0)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def time_left():
            return deadline - time.time() > 2.0

        rng = np.random.default_rng(12345)

        # --- Nelder-Mead Implementation ---
        def nelder_mead(x0, f0, max_iter=100):
            nonlocal calls
            # Simplex setup
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            # Initial simplex: x0 and x0 + delta * e_i
            delta = 0.1
            simplex = [x0.copy()]
            for i in range(dim):
                x_i = x0.copy()
                x_i[i] += delta
                simplex.append(x_i)
            
            # Evaluate initial simplex
            f_values = [f0]
            for i in range(1, dim + 1):
                if not time_left() or calls >= budget:
                    return x0, f0
                f_i = func(simplex[i])
                calls += 1
                f_values.append(f_i)
                update_best(simplex[i], f_i)
            
            # Reorder simplex
            for i in range(dim + 1):
                min_idx = i + np.argmin(f_values[i:])
                simplex[i], simplex[min_idx] = simplex[min_idx], simplex[i]
                f_values[i], f_values[min_idx] = f_values[min_idx], f_values[i]

            for _ in range(max_iter):
                if not time_left() or calls >= budget:
                    break
                
                # Centroid of best dim points
                centroid = np.mean(simplex[:dim], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[dim])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1
                update_best(x_r, f_r)

                if f_values[0] <= f_r < f_values[-1]:
                    # Accept reflection
                    simplex[dim] = x_r
                    f_values[dim] = f_r
                elif f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1
                    update_best(x_e, f_e)
                    
                    if f_e < f_r:
                        simplex[dim] = x_e
                        f_values[dim] = f_e
                    else:
                        simplex[dim] = x_r
                        f_values[dim] = f_r
                else:
                    # Contraction
                    if f_r < f_values[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    update_best(x_c, f_c)

                    if f_c < f_values[-1]:
                        simplex[dim] = x_c
                        f_values[dim] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not time_left() or calls >= budget:
                                break
                            f_values[i] = func(simplex[i])
                            calls += 1
                            update_best(simplex[i], f_values[i])
                        if not time_left() or calls >= budget:
                            break

                # Reorder simplex
                for i in range(dim + 1):
                    min_idx = i + np.argmin(f_values[i:])
                    simplex[i], simplex[min_idx] = simplex[min_idx], simplex[i]
                    f_values[i], f_values[min_idx] = f_values[min_idx], f_values[i]

            return simplex[0], f_values[0]

        # --- Phase 1: Multi-Start Nelder-Mead ---
        # Use multiple random starting points to explore the landscape
        n_starts = 10  # Number of random starts
        
        for start_idx in range(n_starts):
            if not time_left() or calls >= budget:
                break
            
            # Generate random start point
            x_start = rng.uniform(-4, 4, dim)
            f_start = func(x_start)
            calls += 1
            update_best(x_start, f_start)
            
            # Run Nelder-Mead from this start
            # Limit iterations based on remaining budget
            remaining_budget = budget - calls
            max_iter = min(50, max(10, remaining_budget // 10))
            
            x_opt, f_opt = nelder_mead(x_start, f_start, max_iter)
            update_best(x_opt, f_opt)

        # --- Phase 2: Final Refinement around Best Found ---
        if time_left() and calls < budget:
            x_refine = best_x.copy()
            f_refine = best_f
            # Use a smaller step size for final polish
            delta = 0.01
            simplex = [x_refine.copy()]
            for i in range(dim):
                x_i = x_refine.copy()
                x_i[i] += delta
                simplex.append(x_i)
            
            f_values = [f_refine]
            for i in range(1, dim + 1):
                if not time_left() or calls >= budget:
                    break
                f_i = func(simplex[i])
                calls += 1
                f_values.append(f_i)
                update_best(simplex[i], f_i)
            
            # Run a few more Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            for _ in range(20):
                if not time_left() or calls >= budget:
                    break
                
                centroid = np.mean(simplex[:dim], axis=0)
                x_r = centroid + alpha * (centroid - simplex[dim])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1
                update_best(x_r, f_r)

                if f_values[0] <= f_r < f_values[-1]:
                    simplex[dim] = x_r
                    f_values[dim] = f_r
                elif f_r < f_values[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1
                    update_best(x_e, f_e)
                    
                    if f_e < f_r:
                        simplex[dim] = x_e
                        f_values[dim] = f_e
                    else:
                        simplex[dim] = x_r
                        f_values[dim] = f_r
                else:
                    if f_r < f_values[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    update_best(x_c, f_c)

                    if f_c < f_values[-1]:
                        simplex[dim] = x_c
                        f_values[dim] = f_c
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not time_left() or calls >= budget:
                                break
                            f_values[i] = func(simplex[i])
                            calls += 1
                            update_best(simplex[i], f_values[i])
                        if not time_left() or calls >= budget:
                            break

                for i in range(dim + 1):
                    min_idx = i + np.argmin(f_values[i:])
                    simplex[i], simplex[min_idx] = simplex[min_idx], simplex[i]
                    f_values[i], f_values[min_idx] = f_values[min_idx], f_values[i]

        return best_x, best_f

# EVOLVE-BLOCK-END
