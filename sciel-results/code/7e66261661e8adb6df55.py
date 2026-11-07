import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s/330s kill
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        def clip(v):
            return np.clip(np.asarray(v, dtype=float), -5.0, 5.0)
        
        def time_ok():
            return time.time() < deadline

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        def call_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget or not time_ok():
                return np.inf
            calls_used += 1
            x_c = clip(x)
            f_val = func(x_c)
            if f_val < best_f:
                best_f = f_val
                best_x = x_c.copy()
            return f_val

        def run_nelder_mead(center, initial_step, max_iter_nm=100):
            nonlocal best_x, best_f, calls_used

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            # Initialize simplex
            simplex = [clip(center).copy()]
            simplex_f = [call_func(simplex[0])]
            
            if calls_used >= budget - 5:
                return

            # Generate random orthogonal-ish directions for the rest of the simplex
            # This is critical for rotated ridges (f13)
            for i in range(dim):
                if calls_used >= budget - 5:
                    return
                direction = rng.normal(0, 1, dim)
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    direction = np.zeros(dim)
                    direction[i] = 1.0
                else:
                    direction /= norm
                x_i = center + initial_step * direction
                x_i = clip(x_i)
                f_i = call_func(x_i)
                simplex.append(x_i)
                simplex_f.append(f_i)

            # Main NM loop
            for _ in range(max_iter_nm):
                if not time_ok() or calls_used >= budget - 5:
                    break

                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check for convergence
                if max(simplex_f) - min(simplex_f) < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]

                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = clip(x_r)
                f_r = call_func(x_r)

                if calls_used >= budget - 5:
                    break

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = clip(x_e)
                    f_e = call_func(x_e)

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
                        f_c = call_func(x_c)

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
                                f_s = call_func(simplex[i])
                                simplex_f[i] = f_s
                    else:
                        # Inner contraction
                        x_c = centroid - rho * (centroid - worst)
                        x_c = clip(x_c)
                        f_c = call_func(x_c)

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
                                f_s = call_func(simplex[i])
                                simplex_f[i] = f_s

        # Phase 1: Exploration with randomized simplex
        # Use ~80% of budget for the main NM run
        # We start with a reasonably large step to cover the box [-5, 5]
        # f13 minimizer is in [-4, 4]. A step of 2.0 is a good start.
        
        # We can do a few restarts if the first one doesn't converge well or uses too few calls
        # But NM usually uses most of its max_iter if not converged.
        
        # Let's use a single robust NM run with a large enough max_iter to consume budget,
        # or split budget into 2 runs if the first one stagnates early.
        
        # Strategy: 
        # 1. Start NM from a random point in [-3, 3] with step 2.0.
        # 2. If it finishes early (converged or stuck) and we have > 20% budget left, 
        #    restart from the best point found with a smaller step for polishing.
        
        x_start = rng.uniform(-3, 3, dim)
        
        # Run NM with a large max_iter, it will stop when budget is hit or converged
        run_nelder_mead(x_start, initial_step=2.0, max_iter_nm=500)
        
        # Phase 2: Polishing
        # If we have calls left, do a local search from best_x
        if calls_used < budget - 10 and time_ok():
            # Use a smaller step for local refinement
            # The ridge is sharp, so we need to be careful not to jump off it.
            # A step of 0.5 is reasonable for polishing.
            run_nelder_mead(best_x, initial_step=0.5, max_iter_nm=100)
            
            # If still calls left, try even smaller step
            if calls_used < budget - 10 and time_ok():
                run_nelder_mead(best_x, initial_step=0.1, max_iter_nm=50)

        return best_x, best_f
