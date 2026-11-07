import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # 1. Initial Exploration
        # Use a few random starts to find a good basin
        n_starts = 10
        best_x = None
        best_f = np.inf
        calls = 0

        for i in range(n_starts):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # 2. Main Optimization: (1+10)-ES
        sigma = 1.0
        lambda_pop = 10
        no_improve_gens = 0
        max_no_improve_gens = 10
        restarts = 0
        max_restarts = 5

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            for _ in range(lambda_pop):
                # Check time before each evaluation to be safe
                if time.time() > deadline or calls >= budget:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                offspring.append((fy, y))

            if not offspring:
                break

            # Find best offspring
            best_off_f, best_off_x = min(offspring, key=lambda t: t[0])
            
            # Check for improvement
            improved = False
            if best_off_f < best_f:
                best_f = best_off_f
                best_x = best_off_x
                improved = True
                no_improve_gens = 0
            else:
                no_improve_gens += 1

            # Adapt step size
            # Count successes in this generation
            successes = sum(1 for f, x in offspring if f < best_f + 1e-12) # Note: best_f might have updated, so this is slightly off.
            # Better: compare to the previous best_f
            prev_best_f = best_f
            if improved:
                # Success rate based on how many offspring were better than the OLD best
                successes = sum(1 for f, x in offspring if f < prev_best_f)
            else:
                successes = 0

            success_rate = successes / len(offspring)
            
            if success_rate > 0.3:
                sigma *= 1.2
            elif success_rate < 0.1:
                sigma *= 0.8
            else:
                sigma *= 1.0

            # Bounds for sigma
            sigma = max(sigma, 1e-6)
            sigma = min(sigma, 10.0)

            # Restart logic
            if no_improve_gens >= max_no_improve_gens and restarts < max_restarts:
                restarts += 1
                no_improve_gens = 0
                # Try a random restart
                x_new = rng.uniform(-5, 5, dim)
                fx_new = func(x_new)
                calls += 1
                if fx_new < best_f:
                    best_f = fx_new
                    best_x = x_new.copy()
                    sigma = 2.0
                else:
                    # Stay at best but increase sigma to escape
                    sigma = max(sigma * 2.0, 1.0)
                
                # Increase patience for next restart
                max_no_improve_gens = int(max_no_improve_gens * 1.5)

        # 3. Final Polish
        # If we have budget left and sigma is small, do coordinate-wise refinement
        if calls < budget and sigma < 0.1:
            polish_steps = min(budget - calls, 50)
            step = 0.01
            for _ in range(polish_steps):
                if time.time() > deadline or calls >= budget:
                    break
                improved = False
                for i in range(dim):
                    for delta in [step, -step]:
                        x_try = best_x.copy()
                        x_try[i] += delta
                        x_try = np.clip(x_try, -5, 5)
                        if np.all(x_try == best_x):
                            continue
                        fy = func(x_try)
                        calls += 1
                        if fy < best_f:
                            best_f = fy
                            best_x = x_try.copy()
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-5:
                        break

        return best_x, best_f
