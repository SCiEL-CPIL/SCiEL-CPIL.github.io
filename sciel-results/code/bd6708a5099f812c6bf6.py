"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, time guard, and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # 325s deadline, 5s safety margin before 330s kill
        rng = np.random.default_rng(0)

        # Best-so-far tracking
        best_x = np.zeros(self.dim)
        best_fx = np.inf
        total_calls = 0

        def check_time():
            return time.time() < deadline

        # Initial exploration: try a few random starts to find a good region
        # Use a small number of initial points to get a rough landscape sense
        n_init = min(10, self.budget // 5)
        for _ in range(n_init):
            if total_calls >= self.budget or not check_time():
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            total_calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if total_calls >= self.budget or not check_time():
            return best_x, best_fx

        # Now run (1+1)-ES from the best point found
        x = best_x.copy()
        fx = best_fx
        sigma = 1.5  # Start with moderate step size

        # Adaptive restart mechanism: if no improvement for a while, restart from best
        no_improve_count = 0
        max_no_improve = 50
        restart_count = 0
        max_restarts = 5

        while total_calls < self.budget and check_time():
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            total_calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                no_improve_count = 0
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                sigma *= 0.9
                no_improve_count += 1

            # Adaptive restart: if stuck for too long, try restarting from best
            if no_improve_count >= max_no_improve:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                # Restart from best point with a larger sigma
                x = best_x.copy()
                fx = best_fx
                sigma = 3.0
                no_improve_count = 0
                max_no_improve = max(20, max_no_improve // 2)

            # Gradually increase patience as we get closer to optimum
            if no_improve_count == 0 and sigma < 0.01:
                # Very small sigma, might be at optimum or stuck in flat region
                # Try a few larger steps to escape
                for _ in range(5):
                    if total_calls >= self.budget or not check_time():
                        break
                    y = np.clip(x + 0.5 * rng.standard_normal(self.dim), -5, 5)
                    fy = func(y)
                    total_calls += 1
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
