"""Hybrid (1+1)-ES global exploration + Nelder-Mead local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer combining (1+1)-ES for global exploration 
    and Nelder-Mead for local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s/330s limit
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return calls < budget

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # --- Phase 1: (1+1)-ES for Global Exploration ---
        # Use ~60% of the budget for exploration to ensure we find the correct basin
        es_budget = int(budget * 0.6)
        es_calls = 0
        
        # Initialize at center or a random point? Center is often good for symmetric functions, 
        # but the task mentions hidden shifts. Random is safer against specific shifts.
        x = rng.uniform(-4, 4, dim)
        f_x = func(x)
        calls += 1
        record(x, f_x)

        sigma = 1.5  # Initial step size
        no_improve = 0
        max_no_improve = 25

        while es_calls < es_budget and time_ok() and budget_ok():
            # Generate mutant
            step = rng.standard_normal(dim) * sigma
            x_new = x + step
            x_new = np.clip(x_new, -5, 5)
            
            f_new = func(x_new)
            calls += 1
            es_calls += 1
            record(x_new, f_new)

            if f_new <= f_x:
                # Success
                x = x_new
                f_x = f_new
                no_improve = 0
                # Increase step size on success
                sigma *= 1.1
            else:
                # Failure
                no_improve += 1
                # Decrease step size on failure
                sigma *= 0.95

            # Clamp sigma
            sigma = np.clip(sigma, 0.01, 10.0)

            # Restart if stuck
            if no_improve > max_no_improve:
                x = rng.uniform(-4, 4, dim)
                f_x = func(x)
                calls += 1
                es_calls += 1
                record(x, f_x)
                sigma = 1.5
                no_improve = 0

        # --- Phase 2: Nelder-Mead Refinement ---
        # Use the remaining budget for local refinement from the best point found
        if best_x is not None and time_ok() and budget_ok():
            x0 = best_x.copy()
            f0 = best_f
            
            # Build initial simplex
            # Small initial step to stay close to the good region found by ES
            alpha_init = 0.2 
            simplex = [x0.copy()]
            f_simplex = [f0]
            
            # We have already evaluated x0, so we don't count it again in the loop logic 
            # but we need to build the simplex. The simplex construction costs 'dim' evaluations.
            # Ensure we have budget for simplex construction.
            if calls + dim <= budget and time_ok():
                for i in range(dim):
                    x_i = x0.copy()
                    # Perturb dimension i
                    # Choose step direction away from boundary if possible
                    if x0[i] < 4.0:
                        x_i[i] += alpha_init
                    else:
                        x_i[i] -= alpha_init
                    x_i = np.clip(x_i, -5, 5)
                    f_i = func(x_i)
                    calls += 1
                    simplex.append(x_i)
                    f_simplex.append(f_i)
                    record(x_i, f_i)
                
                # Nelder-Mead parameters
                gamma = 1.0   # Reflection
                rho = 2.0     # Expansion
                sigma_nm = 0.5 # Contraction
                shrink = 0.5

                while time_ok() and budget_ok():
                    # Sort simplex by function value
                    indices = np.argsort(f_simplex)
                    simplex = [simplex[i] for i in indices]
                    f_simplex = [f_simplex[i] for i in indices]

                    x_best = simplex[0]
                    f_best = f_simplex[0]
                    x_worst = simplex[-1]
                    f_worst = f_simplex[-1]
                    x_second_worst = simplex[-2]
                    f_second_worst = f_simplex[-2]

                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    x_r = centroid + gamma * (centroid - x_worst)
                    x_r = np.clip(x_r, -5, 5)
                    if not time_ok() or not budget_ok():
                        break
                    f_r = func(x_r)
                    calls += 1
                    record(x_r, f_r)

                    if f_r < f_best:
                        # Expansion
                        x_e = centroid + rho * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        if not time_ok() or not budget_ok():
                            break
                        f_e = func(x_e)
                        calls += 1
                        record(x_e, f_e)
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_simplex[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            f_simplex[-1] = f_r
                    elif f_r < f_second_worst:
                        # Reflection accepted
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                    else:
                        # Contraction
                        if f_r < f_worst:
                            # Outside contraction
                            x_c = centroid + sigma_nm * (x_r - centroid)
                        else:
                            # Inside contraction
                            x_c = centroid - sigma_nm * (x_worst - centroid)
                        x_c = np.clip(x_c, -5, 5)

                        if not time_ok() or not budget_ok():
                            break
                        f_c = func(x_c)
                        calls += 1
                        record(x_c, f_c)

                        if f_c < min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                if not time_ok() or not budget_ok():
                                    break
                                simplex[i] = x_best + shrink * (simplex[i] - x_best)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_simplex[i] = func(simplex[i])
                                calls += 1
                                record(simplex[i], f_simplex[i])
                            
                            if not time_ok() or not budget_ok():
                                break

                    # Check for convergence
                    f_spread = np.max(f_simplex) - np.min(f_simplex)
                    if f_spread < 1e-12:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
