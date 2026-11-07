"""Derivative-free optimizer for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate descent with adaptive step sizes and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill
        calls_made = 0
        budget = self.budget
        dim = self.dim

        # Track best solution found
        best_x = None
        best_f = np.inf

        def check_deadline():
            return time.time() >= deadline

        def count_call():
            nonlocal calls_made
            calls_made += 1
            return calls_made > budget or check_deadline()

        # Multi-restart coordinate descent
        n_restarts = 5
        budget_per_restart = max(1, budget // n_restarts)

        for restart in range(n_restarts):
            if check_deadline() or calls_made >= budget:
                break

            # Initialize: random point, biased toward center for later restarts
            if restart == 0:
                rng = np.random.default_rng(42)
                x = rng.uniform(-3, 3, dim)
            else:
                rng = np.random.default_rng(42 + restart)
                spread = 3.0 - 0.3 * restart
                x = rng.uniform(-spread, spread, dim)

            fx = func(x)
            count_call()
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            if check_deadline() or calls_made >= budget:
                break

            # Adaptive step sizes per dimension
            steps = np.ones(dim) * 1.0
            no_improve_count = 0
            max_no_improve = 10

            # Coordinate descent loop
            while calls_made < budget and not check_deadline():
                improved = False
                for i in range(dim):
                    if calls_made >= budget or check_deadline():
                        break

                    # Try positive and negative steps
                    best_local_f = fx
                    best_local_x = x.copy()
                    step = steps[i]

                    # Line search along dimension i
                    for direction in [1.0, -1.0]:
                        for _ in range(5):  # up to 5 line search steps
                            x_try = x.copy()
                            x_try[i] += direction * step
                            x_try[i] = np.clip(x_try[i], -5, 5)

                            f_try = func(x_try)
                            count_call()
                            if calls_made >= budget or check_deadline():
                                break

                            if f_try < best_local_f:
                                best_local_f = f_try
                                best_local_x = x_try.copy()
                                step *= 1.5  # expand step
                            else:
                                step *= 0.5  # shrink step
                                break

                        if calls_made >= budget or check_deadline():
                            break

                    if best_local_f < fx:
                        x = best_local_x
                        fx = best_local_f
                        improved = True
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        steps[i] *= 1.2  # slightly increase step for good dimension
                    else:
                        steps[i] *= 0.7  # decrease step for unimproved dimension

                if not improved:
                    no_improve_count += 1
                    if no_improve_count >= max_no_improve:
                        break
                else:
                    no_improve_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
