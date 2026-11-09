"""Optimizer for BBOB f4 (Büche-Rastrigin) on [-5,5]^5."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls_used = 1

        # (1+lambda)-ES with adaptation
        sigma = 3.0
        lam = 16
        # 1/5 success rule adaptation
        success_counter = 0
        total_counter = 0
        adaptation_period = 10

        def time_ok():
            return time.time() < deadline

        while calls_used < budget and time_ok():
            # Generate offspring
            offspring = []
            offspring_f = []
            for i in range(lam):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                offspring.append(y)
                offspring_f.append(fy)

            if not offspring:
                break

            # Select best offspring
            best_idx = int(np.argmin(offspring_f))
            best_off_y = offspring[best_idx]
            best_off_f = offspring_f[best_idx]

            # Update success counter
            success_counter += int(best_off_f < fx)
            total_counter += len(offspring)

            # Adapt sigma using 1/5 rule
            if total_counter >= adaptation_period:
                success_rate = success_counter / total_counter
                if success_rate > 0.5:
                    sigma *= 1.5
                elif success_rate < 0.1:
                    sigma *= 0.7
                else:
                    sigma *= 0.9
                success_counter = 0
                total_counter = 0

            # Update parent
            if best_off_f < fx:
                x = best_off_y
                fx = best_off_f
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            # Safety: ensure sigma stays in reasonable range
            sigma = np.clip(sigma, 1e-4, 10.0)

        # Final polish: coordinate descent with shrinking steps
        if time_ok() and calls_used < budget:
            x = best_x.copy()
            fx = best_f
            for step_scale in [0.5, 0.1, 0.01, 0.001, 0.0001]:
                if not time_ok() or calls_used >= budget:
                    break
                for _ in range(3):  # multiple passes per scale
                    if not time_ok() or calls_used >= budget:
                        break
                    improved = False
                    for d in range(dim):
                        for direction in [-1, 1]:
                            if not time_ok() or calls_used >= budget:
                                break
                            trial = x.copy()
                            trial[d] = np.clip(x[d] + direction * step_scale, -5, 5)
                            if np.all(trial == x):
                                continue
                            f_trial = func(trial)
                            calls_used += 1
                            if f_trial < fx:
                                x = trial
                                fx = f_trial
                                improved = True
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
        return best_x, best_f
