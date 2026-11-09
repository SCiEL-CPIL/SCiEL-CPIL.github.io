"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with fixed step-size hill-climb per restart."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 315.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Best solution found overall
        best_x = None
        best_f = np.inf
        eval_count = 0

        def try_update(x, fx):
            nonlocal best_x, best_f, eval_count
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Multi-start strategy: partition budget into 30 restarts of 10 evals each
        # Each restart: 1 initial eval + 9 hill-climb steps
        # Use fixed step size per restart (no global adaptation)
        num_restarts = 30
        evals_per_restart = 10  # 1 initial + 9 steps

        # Pre-generate starting points for diversity
        # Mix of random points and perturbations of origin/best
        start_points = []
        for i in range(num_restarts):
            if i == 0:
                # Start at origin (good for many BBOB functions)
                start_points.append(np.zeros(n))
            elif i == 1:
                # Start at a random point in [-3, 3]
                start_points.append(rng.uniform(-3, 3, n))
            elif i == 2:
                # Start at a random point in [-4, 4]
                start_points.append(rng.uniform(-4, 4, n))
            else:
                # Mix: some near origin, some random
                if i % 3 == 0:
                    start_points.append(rng.uniform(-2, 2, n))
                elif i % 3 == 1:
                    start_points.append(rng.uniform(-3, 3, n))
                else:
                    start_points.append(rng.uniform(-4, 4, n))

        # Fixed step sizes for each restart (slightly varied for diversity)
        base_sigma = 1.5

        for restart_idx in range(num_restarts):
            # Check budget and time
            if eval_count >= budget:
                break
            if time.time() > deadline - 3.0:
                break

            # Determine how many evals we can use for this restart
            remaining = budget - eval_count
            steps_this_restart = min(evals_per_restart, remaining)
            if steps_this_restart < 2:
                break

            # Get starting point
            if restart_idx < len(start_points):
                x = start_points[restart_idx].copy()
            else:
                # If we run out of pre-generated points, use random
                x = rng.uniform(-4, 4, n)

            # Evaluate starting point
            fx = func(x)
            try_update(x, fx)

            if eval_count >= budget:
                break

            # Fixed step-size hill-climb (steepest descent-like with fixed sigma)
            # Use a fixed sigma for this restart
            sigma = base_sigma * (1.0 + 0.1 * (restart_idx % 5))  # Slight variation

            for step in range(steps_this_restart - 1):  # -1 because we already used 1 eval
                if eval_count >= budget:
                    break
                if time.time() > deadline - 3.0:
                    break

                # Generate random direction
                z = rng.standard_normal(n)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                eval_count += 1

                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                # Note: we do NOT adapt sigma within a restart (fixed step size)
                # This is the key structural difference from the parent

        return best_x, best_f
# EVOLVE-BLOCK-END
