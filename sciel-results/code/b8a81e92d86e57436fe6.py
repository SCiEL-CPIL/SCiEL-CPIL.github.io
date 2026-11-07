"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derandomized Discrete PSO with ring topology and adaptive inertia."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        rng = np.random.default_rng(123)
        dim = self.dim
        budget = self.budget

        # Swarm size: small to fit within budget
        # With 500 calls and dim=5, we can have a swarm of ~10-20
        # Let's use 10 particles, each evaluated once per generation
        # 500 / 10 = 50 generations
        n_particles = 10
        n_generations = budget // n_particles
        if n_generations < 1:
            n_particles = 1
            n_generations = budget

        # Initialize particles uniformly in [-5, 5]^dim
        positions = rng.uniform(-5, 5, (n_particles, dim))
        velocities = np.zeros((n_particles, dim))

        # Evaluate initial positions
        f_values = np.array([func(p) for p in positions])

        # Personal bests
        pbest_pos = positions.copy()
        pbest_val = f_values.copy()

        # Global best
        gbest_idx = np.argmin(f_values)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_val = pbest_val[gbest_idx]

        # Track best ever
        best_ever_val = gbest_val
        best_ever_pos = gbest_pos.copy()

        # Velocity bound
        v_max = 1.0

        # Inertia weight: linear decrease from 0.9 to 0.4
        w_start = 0.9
        w_end = 0.4

        # Cognitive and social coefficients
        c1 = 1.5
        c2 = 1.5

        for gen in range(n_generations):
            # Check time
            if time.time() > deadline:
                break

            # Compute inertia weight for this generation
            if n_generations > 1:
                progress = gen / (n_generations - 1)
            else:
                progress = 0.0
            w = w_start + (w_end - w_start) * progress

            for i in range(n_particles):
                if time.time() > deadline:
                    break

                # Derandomized: use fixed coefficients instead of random
                # v = w*v + c1*(pbest - x) + c2*(gbest - x)
                r1 = 0.5  # derandomized
                r2 = 0.5  # derandomized

                velocities[i] = (w * velocities[i]
                                + c1 * r1 * (pbest_pos[i] - positions[i])
                                + c2 * r2 * (gbest_pos - positions[i]))

                # Clip velocity to [-v_max, v_max]
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                # Update position
                positions[i] = positions[i] + velocities[i]

                # Clip to box [-5, 5]
                positions[i] = np.clip(positions[i], -5, 5)

                # Evaluate
                f_new = func(positions[i])

                # Update personal best
                if f_new < pbest_val[i]:
                    pbest_val[i] = f_new
                    pbest_pos[i] = positions[i].copy()

                    # Update global best
                    if f_new < gbest_val:
                        gbest_val = f_new
                        gbest_pos = positions[i].copy()
                        best_ever_val = f_new
                        best_ever_pos = positions[i].copy()

        return best_ever_pos, best_ever_val
# EVOLVE-BLOCK-END
