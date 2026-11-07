"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Particle Swarm Optimization with inertia weight decay."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # PSO parameters
        n_particles = 20
        w_start = 0.9
        w_end = 0.4
        c1 = 2.0
        c2 = 2.0

        # Initialize particles uniformly in [-5, 5]^dim
        particles = rng.uniform(-5, 5, (n_particles, dim))
        particles = np.clip(particles, -5, 5)
        velocities = rng.uniform(-1, 1, (n_particles, dim))

        # Evaluate initial particles
        fvals = np.array([func(p) for p in particles])
        personal_best = particles.copy()
        personal_best_val = fvals.copy()
        global_best_idx = np.argmin(fvals)
        global_best = particles[global_best_idx].copy()
        global_best_val = fvals[global_best_idx]

        # Track best for return
        best_x = global_best.copy()
        best_f = global_best_val

        calls_made = n_particles

        while calls_made < budget:
            if time.time() >= deadline:
                break

            # Inertia weight decay
            progress = min(1.0, calls_made / budget)
            w = w_start + (w_end - w_start) * progress

            r1 = rng.random((n_particles, dim))
            r2 = rng.random((n_particles, dim))

            # Update velocities
            velocities = (w * velocities +
                          c1 * r1 * (personal_best - particles) +
                          c2 * r2 * (global_best - particles))

            # Update positions
            particles = particles + velocities
            particles = np.clip(particles, -5, 5)

            # Evaluate
            fvals = np.array([func(p) for p in particles])
            calls_made += n_particles

            # Update personal bests
            improved = fvals < personal_best_val
            personal_best[improved] = particles[improved]
            personal_best_val[improved] = fvals[improved]

            # Update global best
            idx = np.argmin(fvals)
            if fvals[idx] < global_best_val:
                global_best_val = fvals[idx]
                global_best = particles[idx].copy()
                best_x = global_best.copy()
                best_f = global_best_val

        return best_x, best_f
# EVOLVE-BLOCK-END
