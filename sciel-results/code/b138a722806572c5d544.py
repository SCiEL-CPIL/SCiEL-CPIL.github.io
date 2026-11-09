"""Optimized solver for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid random sampling + (1+1)-ES with 1/5 rule and sigma reset on stagnation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 330s kill
        
        # Call counter to strictly respect budget
        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)
        rng = np.random.default_rng(42)

        def counting_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Random sampling to find a good basin
        n_random = min(50, budget // 10)
        for _ in range(n_random):
            if time.time() > deadline or calls >= budget:
                break
            x = rng.uniform(-5, 5, size=dim)
            counting_func(x)

        # Phase 2: (1+1)-ES with 1/5 rule
        # Start from best point found
        x = best_x.copy()
        sigma = 1.0  # Initial step size
        stagnation_window = 10  # Number of failed trials before sigma reset
        no_improve_count = 0
        best_f_es = best_f
        
        while calls < budget and time.time() < deadline:
            # Generate trial point
            direction = rng.standard_normal(dim)
            trial_x = x + sigma * direction
            trial_x = np.clip(trial_x, -5, 5)
            
            f_trial = counting_func(trial_x)
            
            if f_trial < best_f_es:
                # Success: move to trial point, increase sigma slightly
                x = trial_x.copy()
                best_f_es = f_trial
                no_improve_count = 0
                sigma *= 1.05  # Slight increase on success
            else:
                # Failure: keep current x, track stagnation
                no_improve_count += 1
                if no_improve_count >= stagnation_window:
                    # Reset sigma to explore more
                    sigma = max(0.1, sigma * 0.5)
                    no_improve_count = 0
                    # Occasionally restart from a random point if stuck
                    if sigma < 0.05 and calls < budget - 20:
                        x = rng.uniform(-5, 5, size=dim)
                        best_f_es = counting_func(x)
                        sigma = 1.0
                        no_improve_count = 0
            
            # 1/5 rule: if success rate > 1/5, increase sigma; else decrease
            # We approximate by checking if we've had recent successes
            # Simple heuristic: adjust based on no_improve_count
            if no_improve_count == 0:
                # Recent success, keep or slightly increase
                pass
            elif no_improve_count < stagnation_window:
                # Some failures, slightly decrease sigma
                sigma *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
