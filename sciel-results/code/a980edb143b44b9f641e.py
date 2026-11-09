"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start optimizer with adaptive (1+lambda)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s kill

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Multi-start coarse exploration
        # Weierstrass has many local minima; sampling broadly helps find the global basin
        # Use stratified sampling for better coverage
        n_init = min(30, budget // 8)
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            # Mix of broad and moderate sampling
            if i < n_init // 3:
                x = rng.uniform(-4.5, 4.5, dim)
            elif i < 2 * n_init // 3:
                x = rng.uniform(-3.0, 3.0, dim)
            else:
                x = rng.uniform(-1.5, 1.5, dim)
            evaluate(x)

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Population-based local refinement from best point
        # Use (1+lambda)-ES with adaptive step size
        x = best_x.copy() if best_x is not None else rng.uniform(-1, 1, dim)
        fx = best_f

        sigma = 1.5  # Start with larger initial step for exploration
        pop_size = 8
        successes = []
        window_size = 10
        stagnation = 0
        max_stagnation = 30
        last_improvement_call = calls

        while calls < budget and time.time() < deadline:
            # Generate population of offspring
            improvements = 0
            for j in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break

                # Use different scales for diversity in offspring
                scale = sigma
                if j % 3 == 1:
                    scale = sigma * 0.5
                elif j % 3 == 2:
                    scale = sigma * 1.5

                y = x + scale * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)

                if fy < fx:
                    improvements += 1
                    x, fx = y, fy
                    successes.append(1)
                    stagnation = 0
                    last_improvement_call = calls
                else:
                    successes.append(0)
                    stagnation += 1

            # Trim success window
            if len(successes) > window_size:
                successes = successes[-window_size:]

            # Adaptive sigma: use success rate with smoother adaptation
            if len(successes) >= window_size:
                rate = sum(successes) / window_size
                if rate > 0.30:
                    sigma *= 1.20
                elif rate > 0.15:
                    sigma *= 1.05
                elif rate > 0.05:
                    sigma *= 0.95
                else:
                    sigma *= 0.75

            # Clamp sigma
            sigma = np.clip(sigma, 0.003, 5.0)

            # Restart mechanism: if stuck, try larger jumps or reset
            if stagnation > max_stagnation:
                if sigma < 0.10:
                    # Very small step: likely in a sharp local min, restart with larger sigma
                    x = best_x.copy()
                    fx = best_f
                    sigma = 2.5
                    stagnation = 0
                    successes = []
                else:
                    # Moderate step: try a few larger jumps to escape plateau
                    escaped = False
                    for _ in range(min(8, budget - calls)):
                        if calls >= budget or time.time() > deadline:
                            break
                        y = x + sigma * 2.5 * rng.standard_normal(dim)
                        y = np.clip(y, -5, 5)
                        fy = evaluate(y)
                        if fy < fx:
                            x, fx = y, fy
                            stagnation = 0
                            escaped = True
                            break
                    if not escaped:
                        # No improvement from jumps, shrink and continue
                        sigma *= 0.6
                        stagnation = 0
                    successes = []

            # If we've made progress recently, reduce stagnation counter
            if calls - last_improvement_call < pop_size:
                stagnation = min(stagnation, pop_size)

        # Phase 3: Fine-tuning with small steps - use remaining budget more effectively
        if calls < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            fine_sigma = 0.08
            fine_window = 8
            fine_successes = []

            remaining = budget - calls
            # Use up to 40% of remaining budget for fine-tuning, at least 20 calls
            fine_budget = max(20, min(remaining // 2, 80))

            for _ in range(fine_budget):
                if calls >= budget or time.time() > deadline:
                    break
                y = x + fine_sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    fine_sigma *= 1.08
                    fine_successes.append(1)
                else:
                    fine_sigma *= 0.92
                    fine_successes.append(0)

                if len(fine_successes) > fine_window:
                    fine_successes = fine_successes[-fine_window:]

                # Adapt fine_sigma based on recent success rate
                if len(fine_successes) >= fine_window:
                    rate = sum(fine_successes) / fine_window
                    if rate > 0.4:
                        fine_sigma *= 1.1
                    elif rate < 0.1:
                        fine_sigma *= 0.8

                if fine_sigma < 0.0005:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
