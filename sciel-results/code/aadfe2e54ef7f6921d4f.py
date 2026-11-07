"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simulated Annealing with geometric cooling for Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        total_calls = 1  # already made one call

        # Initial temperature - large to explore globally
        T = 100.0
        # Geometric cooling rate
        cooling_rate = 0.995

        while total_calls < budget:
            if time.time() >= deadline:
                break

            # Generate candidate with step size proportional to temperature
            step_size = max(0.01, T * 0.05)
            y = np.clip(x + step_size * np.random.randn(dim), -5, 5)
            fy = func(y)
            total_calls += 1

            # Update best
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()

            # Simulated annealing acceptance
            delta_f = fy - fx
            if delta_f < 0:
                # Accept improvement
                x = y
                fx = fy
            else:
                # Accept worse solution with probability exp(-delta_f / T)
                if T > 1e-8:
                    accept_prob = np.exp(-delta_f / T)
                    if np.random.random() < accept_prob:
                        x = y
                        fx = fy

            # Cool down
            T *= cooling_rate

        # Final polish: coordinate descent around best_x
        if total_calls < budget and time.time() < deadline - 5:
            polish_x = best_x.copy()
            polish_fx = best_fx
            polish_sigma = 0.1
            polish_budget = min(30, budget - total_calls)
            
            for _ in range(polish_budget):
                if time.time() >= deadline - 2:
                    break
                for d in range(dim):
                    for direction in [1, -1]:
                        trial = polish_x.copy()
                        trial[d] += direction * polish_sigma
                        trial = np.clip(trial, -5, 5)
                        trial_fx = func(trial)
                        total_calls += 1
                        
                        if trial_fx < polish_fx:
                            polish_x = trial
                            polish_fx = trial_fx
                            if polish_fx < best_fx:
                                best_fx = polish_fx
                                best_x = polish_x.copy()
                        if total_calls >= budget or time.time() >= deadline - 2:
                            break
                    if total_calls >= budget or time.time() >= deadline - 2:
                        break
                # Shrink polish step
                polish_sigma *= 0.7

        return best_x, best_fx
# EVOLVE-BLOCK-END
