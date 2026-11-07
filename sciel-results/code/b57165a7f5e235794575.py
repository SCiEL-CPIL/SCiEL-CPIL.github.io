import numpy as np
import time

class Optimizer:
    """
    (1+1)-ES optimizer with adaptive step size and restarts for Weierstrass f16.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 300.0  # Safety margin

        # Global best
        best_x_global = np.zeros(dim)
        best_f_global = np.inf

        # (1+1)-ES parameters
        sigma_init = 2.5
        sigma_min = 1e-7
        sigma_max = 5.0
        
        # Restart logic
        stagnation_limit = 20 # Steps without improvement before restarting

        rng = np.random.default_rng(42)

        evals_used = 0

        def eval_and_update(x):
            nonlocal best_f_global, best_x_global
            fx = func(x)
            if fx < best_f_global:
                best_f_global = fx
                best_x_global = x.copy()
            return fx

        # We will run sessions.
        # Each session starts from a random point (or best point) and runs (1+1)-ES.
        # If the session stagnates, we start a new one.
        
        # Initial point
        x = rng.uniform(-4, 4, size=dim)
        fx = eval_and_update(x)
        evals_used += 1
        best_x_local = x.copy()
        best_f_local = fx
        sigma = sigma_init
        no_improve_count = 0

        while evals_used < budget:
            # Time check
            if time.time() - start_time > time_limit:
                break

            # Generate candidate
            y = rng.standard_normal(dim)
            x_new = best_x_local + sigma * y
            x_new = np.clip(x_new, -5, 5)
            
            fx_new = eval_and_update(x_new)
            evals_used += 1

            if fx_new < best_f_local:
                # Improvement
                best_x_local = x_new.copy()
                best_f_local = fx_new
                no_improve_count = 0
                # Increase sigma slightly to continue exploring
                sigma = min(sigma * 1.1, sigma_max)
            else:
                # No improvement
                no_improve_count += 1
                # Decrease sigma to polish
                sigma = max(sigma * 0.9, sigma_min)

                if no_improve_count >= stagnation_limit:
                    # Restart from a new random point
                    # Or restart from the global best with a large sigma?
                    # Random restart is better for global exploration.
                    x = rng.uniform(-4, 4, size=dim)
                    fx = eval_and_update(x)
                    evals_used += 1
                    if evals_used >= budget:
                        break
                    best_x_local = x.copy()
                    best_f_local = fx
                    sigma = sigma_init
                    no_improve_count = 0

        # Final polish on the global best
        # If we have evals left, do a very fine search around best_x_global
        if evals_used < budget and best_f_global < np.inf:
            x_polish = best_x_global.copy()
            f_polish = best_f_global
            sigma_polish = 0.1
            polish_budget = budget - evals_used
            no_improve_polish = 0
            
            for _ in range(polish_budget):
                if time.time() - start_time > time_limit:
                    break
                y = rng.standard_normal(dim)
                x_try = x_polish + sigma_polish * y
                x_try = np.clip(x_try, -5, 5)
                f_try = func(x_try)
                evals_used += 1
                if f_try < f_polish:
                    x_polish = x_try
                    f_polish = f_try
                    no_improve_polish = 0
                    sigma_polish = min(sigma_polish * 1.1, 0.5)
                else:
                    no_improve_polish += 1
                    sigma_polish = max(sigma_polish * 0.9, 1e-8)
                    if no_improve_polish > 20:
                        break
                
            if f_polish < best_f_global:
                best_f_global = f_polish
                best_x_global = x_polish.copy()

        return best_x_global, best_f_global
