# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Ridge-aware optimizer using Adaptive Nelder-Mead with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(np.asarray(v, dtype=float), -5.0, 5.0)

        def time_ok():
            return time.time() < deadline

        best_x = None
        best_f = np.inf
        calls_used = 0

        def run_adaptive_nm(center, initial_step, max_restarts=3):
            nonlocal best_x, best_f, calls_used

            # Initialize simplex
            simplex = []
            simplex_f = []

            x0 = clip(center)
            f0 = func(x0)
            calls_used += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            simplex.append(x0)
            simplex_f.append(f0)

            if calls_used >= budget - 5:
                return

            # Generate initial simplex
            for i in range(dim):
                if calls_used >= budget - 5:
                    return
                x_i = x0.copy()
                # Use a larger initial step for exploration
                x_i[i] += initial_step
                x_i = clip(x_i)
                f_i = func(x_i)
                calls_used += 1
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
                simplex.append(x_i)
                simplex_f.append(f_i)

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            no_improve_count = 0
            restart_count = 0
            min_f_so_far = min(simplex_f)

            while restart_count < max_restarts:
                if not time_ok() or calls_used >= budget - 10:
                    break
                
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                current_min_f = simplex_f[0]
                
                # Check for convergence
                if max(simplex_f) - min(simplex_f) < 1e-12:
                    break

                # Standard NM Step
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]

                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = clip(x_r)
                f_r = func(x_r)
                calls_used += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if calls_used >= budget - 5:
                    break

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = clip(x_e)
                    f_e = func(x_e)
                    calls_used += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    
                    if calls_used >= budget - 5:
                        break

                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < worst_f:
                        # Outer contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = clip(x_c)
                        f_c = func(x_c)
                        calls_used += 1
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()

                        if calls_used >= budget - 5:
                            break

                        if f_c < f_r:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                if calls_used >= budget - 5:
                                    return
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = clip(simplex[i])
                                f_s = func(simplex[i])
                                calls_used += 1
                                if f_s < best_f:
                                    best_f = f_s
                                    best_x = simplex[i].copy()
                                simplex_f[i] = f_s
                    else:
                        # Inner contraction
                        x_c = centroid - rho * (centroid - worst)
                        x_c = clip(x_c)
                        f_c = func(x_c)
                        calls_used += 1
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()

                        if calls_used >= budget - 5:
                            break

                        if f_c < worst_f:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                if calls_used >= budget - 5:
                                    return
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = clip(simplex[i])
                                f_s = func(simplex[i])
                                calls_used += 1
                                if f_s < best_f:
                                    best_f = f_s
                                    best_x = simplex[i].copy()
                                simplex_f[i] = f_s

                # Check for improvement
                if simplex_f[0] < min_f_so_far - 1e-10:
                    min_f_so_far = simplex_f[0]
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Restart logic: If no improvement, shrink and restart
                if no_improve_count > 20:
                    restart_count += 1
                    # Shrink the simplex around the best point
                    center = simplex[0].copy()
                    step = initial_step * (0.5 ** restart_count)
                    
                    # Re-initialize simplex with smaller step
                    simplex = []
                    simplex_f = []
                    
                    x0 = center
                    f0 = func(x0)
                    calls_used += 1
                    if f0 < best_f:
                        best_f = f0
                        best_x = x0.copy()
                    simplex.append(x0)
                    simplex_f.append(f0)
                    
                    if calls_used >= budget - 5:
                        return

                    for i in range(dim):
                        if calls_used >= budget - 5:
                            return
                        x_i = center.copy()
                        x_i[i] += step
                        x_i = clip(x_i)
                        f_i = func(x_i)
                        calls_used += 1
                        if f_i < best_f:
                            best_f = f_i
                            best_x = x_i.copy()
                        simplex.append(x_i)
                        simplex_f.append(f_i)
                    
                    min_f_so_far = min(simplex_f)
                    no_improve_count = 0

        # Phase 1: Coarse search
        center = np.zeros(dim)
        run_adaptive_nm(center, 3.0)

        # Phase 2: Fine search
        if calls_used < budget - 50 and best_x is not None:
            run_adaptive_nm(best_x, 1.0)

        # Phase 3: Polish
        if calls_used < budget - 20 and best_x is not None:
            run_adaptive_nm(best_x, 0.5)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
