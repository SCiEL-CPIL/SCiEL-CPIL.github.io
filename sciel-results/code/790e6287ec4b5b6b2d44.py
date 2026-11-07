"""Hybrid PSO + (1+1)-ES with coordinate-descent polish for Schwefel f20."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """PSO for global exploration, (1+1)-ES for local refinement, coord-descent polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # Call counter
        calls = [0]

        def F(x):
            calls[0] += 1
            return func(x)

        rng = np.random.default_rng(42)

        # ---- Initial evaluation at center ----
        best_x = np.zeros(dim)
        best_f = F(best_x)

        # ---- Quick random scan for a better starting point ----
        scan_budget = min(15, budget // 15)
        for _ in range(scan_budget):
            if calls[0] >= budget or time.time() >= deadline - 10.0:
                break
            x_try = rng.uniform(-5, 5, dim)
            f_try = F(x_try)
            if f_try < best_f:
                best_f, best_x = f_try, x_try.copy()

        # ---- Phase 1: Particle Swarm Optimization ----
        # PSO parameters
        n_particles = 20
        w_max = 0.9
        w_min = 0.4
        c1 = 1.5  # cognitive coefficient
        c2 = 1.5  # social coefficient

        # Initialize particles around best_x with some spread
        positions = np.array([best_x + rng.normal(0, 1.5, dim) for _ in range(n_particles)])
        positions = np.clip(positions, -5, 5)
        velocities = rng.uniform(-1, 1, (n_particles, dim))

        # Evaluate initial particles
        pbest_positions = positions.copy()
        pbest_values = np.empty(n_particles)
        for i in range(n_particles):
            if calls[0] >= budget or time.time() >= deadline - 10.0:
                break
            pbest_values[i] = F(positions[i])
            if pbest_values[i] < best_f:
                best_f = pbest_values[i]
                best_x = positions[i].copy()
        # Update personal bests
        pbest_positions = positions.copy()
        pbest_values = pbest_values.copy()

        # Global best
        gbest_x = best_x.copy()
        gbest_f = best_f

        # PSO main loop
        max_pso_iters = 30  # limit iterations to leave budget for refinement
        pso_iter = 0
        while pso_iter < max_pso_iters and calls[0] < budget * 0.5 and time.time() < deadline - 15.0:
            # Decay inertia weight
            w = w_max - (w_max - w_min) * (pso_iter / max_pso_iters)

            for i in range(n_particles):
                if calls[0] >= budget or time.time() >= deadline - 15.0:
                    break

                # Update velocity
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                velocities[i] = (w * velocities[i]
                                 + c1 * r1 * (pbest_positions[i] - positions[i])
                                 + c2 * r2 * (gbest_x - positions[i]))

                # Update position
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], -5, 5)

                # Evaluate new position
                f_new = F(positions[i])

                # Update personal best
                if f_new < pbest_values[i]:
                    pbest_values[i] = f_new
                    pbest_positions[i] = positions[i].copy()

                # Update global best
                if f_new < gbest_f:
                    gbest_f = f_new
                    gbest_x = positions[i].copy()
                    if f_new < best_f:
                        best_f = f_new
                        best_x = positions[i].copy()

            pso_iter += 1

        # ---- Phase 2: (1+1)-ES for local refinement ----
        x = gbest_x.copy()
        fx = gbest_f
        if fx < best_f:
            best_f, best_x = fx, x.copy()

        sigma = 0.3
        window_size = 10
        successes = 0
        total_in_window = 0
        stagnant = 0

        while calls[0] < budget * 0.8 and time.time() < deadline - 10.0:
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = F(y)

            if fy < fx:
                x = y
                fx = fy
                successes += 1
                stagnant = 0
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
            else:
                stagnant += 1
                if stagnant > 50:
                    break

            total_in_window += 1
            if total_in_window >= window_size:
                rate = successes / total_in_window
                if rate > 0.2:
                    sigma *= 1.1
                elif rate < 0.2:
                    sigma *= 0.9
                successes = 0
                total_in_window = 0

            sigma = max(1e-8, min(5.0, sigma))
            if sigma < 1e-10:
                break

        # ---- Phase 3: Coordinate-descent polish ----
        x = best_x.copy()
        fx = best_f
        steps = np.full(dim, 0.1)
        no_improve = 0

        while calls[0] < budget and time.time() < deadline - 5.0:
            improved = False
            for d in range(dim):
                if calls[0] >= budget or time.time() >= deadline - 5.0:
                    break
                for sign in [1.0, -1.0]:
                    if calls[0] >= budget or time.time() >= deadline - 5.0:
                        break
                    x_new = x.copy()
                    x_new[d] += sign * steps[d]
                    x_new[d] = np.clip(x_new[d], -5, 5)
                    f_new = F(x_new)
                    if f_new < fx:
                        x = x_new
                        fx = f_new
                        improved = True
                        steps[d] *= 0.8
                        if fx < best_f:
                            best_f, best_x = fx, x.copy()
                    else:
                        steps[d] *= 0.7
                        if steps[d] < 1e-10:
                            steps[d] = 1e-10

            if not improved:
                no_improve += 1
                steps *= 0.5
                if np.all(steps < 1e-12) or no_improve > 15:
                    break
            else:
                no_improve = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
