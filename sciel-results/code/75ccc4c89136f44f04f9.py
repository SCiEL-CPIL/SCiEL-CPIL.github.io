"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based Particle Swarm Optimization (PSO) with inertia damping,
    velocity clamping, and a final (1+1)-ES polish phase for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def time_left():
            return deadline - time.time() > 5.0

        best_x = None
        best_f = np.inf
        calls = 0

        # Reserve budget for the polish phase
        polish_reserve = 80
        main_budget = max(1, budget - polish_reserve)

        # ---- Phase 1: PSO with multi-start initialization ----
        # Initialize a small swarm covering both known Lunacek basin regions
        # and random points to ensure coverage of both basins.
        n_particles = 12

        # Initial positions: mix of structured and random
        init_positions = []
        # Basin 1: near origin
        for _ in range(3):
            init_positions.append(rng.uniform(-1, 1, dim))
        # Basin 2: near shifted point (typical Lunacek shift is around 2.5-3)
        for _ in range(3):
            init_positions.append(2.5 + rng.uniform(-1, 1, dim))
        for _ in range(3):
            init_positions.append(-2.5 + rng.uniform(-1, 1, dim))
        # Random points
        for _ in range(3):
            init_positions.append(rng.uniform(-4, 4, dim))

        positions = np.array(init_positions)
        positions = np.clip(positions, -5, 5)
        velocities = np.zeros((n_particles, dim))

        # Evaluate initial positions
        pbest_positions = np.zeros_like(positions)
        pbest_values = np.full(n_particles, np.inf)
        gbest_position = None
        gbest_value = np.inf

        for i in range(n_particles):
            if not time_left() or calls >= main_budget:
                break
            fx = func(positions[i])
            calls += 1
            pbest_positions[i] = positions[i].copy()
            pbest_values[i] = fx
            if fx < gbest_value:
                gbest_value = fx
                gbest_position = positions[i].copy()
                best_f = fx
                best_x = positions[i].copy()

        # PSO parameters
        c1 = 1.5  # cognitive coefficient
        c2 = 1.5  # social coefficient
        w = 0.9   # inertia weight
        w_min = 0.4
        v_max = 3.0  # max velocity per dimension
        n_iterations = main_budget - calls  # remaining budget for PSO

        for it in range(n_iterations):
            if not time_left() or calls >= main_budget:
                break

            # Decay inertia weight
            w = max(w_min, w * 0.995)

            for i in range(n_particles):
                if not time_left() or calls >= main_budget:
                    break

                # Update velocity
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                velocities[i] = (w * velocities[i]
                                 + c1 * r1 * (pbest_positions[i] - positions[i])
                                 + c2 * r2 * (gbest_position - positions[i]))

                # Clamp velocity
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                # Update position
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], -5, 5)

                # Evaluate
                fy = func(positions[i])
                calls += 1

                if fy < pbest_values[i]:
                    pbest_values[i] = fy
                    pbest_positions[i] = positions[i].copy()
                    if fy < gbest_value:
                        gbest_value = fy
                        gbest_position = positions[i].copy()
                        best_f = fy
                        best_x = positions[i].copy()

            # Periodic diversity injection: if stagnation detected,
            # reset some particles to random locations
            if it > 0 and it % 30 == 0:
                # Check if we've made progress recently
                if gbest_value == pbest_values[np.argmin(pbest_values)]:
                    # No improvement in gbest for a while, inject diversity
                    for j in range(3):
                        idx = rng.integers(0, n_particles)
                        positions[idx] = rng.uniform(-4, 4, dim)
                        velocities[idx] = np.zeros(dim)
                        if not time_left() or calls >= main_budget:
                            break
                        fy = func(positions[idx])
                        calls += 1
                        if fy < pbest_values[idx]:
                            pbest_values[idx] = fy
                            pbest_positions[idx] = positions[idx].copy()
                            if fy < gbest_value:
                                gbest_value = fy
                                gbest_position = positions[idx].copy()
                                best_f = fy
                                best_x = positions[idx].copy()

        # ---- Phase 2: (1+1)-ES polish phase with 1/5 success rule ----
        if best_x is not None and time_left() and calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            sigma_min = 1e-8
            sigma_max = 1.0
            window = 10
            successes_in_window = 0
            total_count = 0
            success_count = 0

            while calls < budget:
                if not time_left():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                total_count += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    successes_in_window += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.05, sigma_max)
                else:
                    sigma = max(sigma * 0.95, sigma_min)

                # 1/5 success rule on a rolling window
                if total_count % window == 0:
                    if successes_in_window >= window / 5.0:
                        sigma = min(sigma * 1.2, sigma_max)
                    else:
                        sigma = max(sigma * 0.8, sigma_min)
                    successes_in_window = 0

                # Stuck detection: boost sigma to escape a local plateau
                if total_count % 50 == 0 and success_count / max(total_count, 1) < 0.1:
                    sigma = max(sigma * 2.0, 0.5)

                # Squeeze harder near the end of the budget
                if calls > budget * 0.9:
                    sigma = max(sigma * 0.98, sigma_min)

        if best_x is None:
            best_x = np.zeros(dim)
            if calls < budget:
                best_f = func(best_x)
                calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
