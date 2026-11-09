"""Particle Swarm Optimization for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """PSO optimizer with adaptive inertia for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # PSO parameters
        n_particles = 20
        w_min, w_max = 0.4, 0.9  # Inertia weight range
        c1, c2 = 1.49445, 1.49445  # Cognitive and social coefficients
        v_max_factor = 0.5  # Max velocity as fraction of box half-size

        # Initialize particles uniformly in [-4, 4]^dim
        positions = rng.uniform(-4, 4, (n_particles, dim))
        velocities = rng.uniform(-2, 2, (n_particles, dim))
        pbests = positions.copy()
        pbest_f = np.full(n_particles, np.inf)

        evals = 0
        gbest_x = None
        gbest_f = np.inf

        # Evaluate initial positions
        for i in range(n_particles):
            if evals >= budget or time.time() > deadline - 5:
                break
            x = np.clip(positions[i], -5, 5)
            f = func(x)
            evals += 1
            pbest_f[i] = f
            pbests[i] = x
            if f < gbest_f:
                gbest_f = f
                gbest_x = x.copy()

        if gbest_x is None:
            return np.zeros(dim), 0.0

        # PSO main loop
        gen = 0
        while evals < budget and time.time() < deadline - 5:
            gen += 1
            # Adapt inertia: decrease over generations
            w = w_max - (w_max - w_min) * min(1.0, gen / 50.0)

            # Update velocities and positions
            for i in range(n_particles):
                if evals >= budget or time.time() > deadline - 5:
                    break

                # Random coefficients
                r1 = rng.random(dim)
                r2 = rng.random(dim)

                # Velocity update: w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                v = w * velocities[i] + c1 * r1 * (pbests[i] - positions[i]) + c2 * r2 * (gbest_x - positions[i])

                # Clamp velocity
                v_max = v_max_factor * 10.0  # box size is 10
                v = np.clip(v, -v_max, v_max)
                velocities[i] = v

                # Position update
                new_pos = positions[i] + v
                new_pos = np.clip(new_pos, -5, 5)
                positions[i] = new_pos

                # Evaluate
                f = func(new_pos)
                evals += 1

                # Update personal best
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbests[i] = new_pos.copy()

                    # Update global best
                    if f < gbest_f:
                        gbest_f = f
                        gbest_x = new_pos.copy()

            # Check for stagnation: if no improvement in last 30 evals, perturb
            if gen % 5 == 0 and evals > 50:
                # Check if we've been stuck
                if gbest_f > 0 and gen > 10:
                    # Add small random perturbation to global best
                    perturbation = 0.1 * rng.standard_normal(dim)
                    candidate = np.clip(gbest_x + perturbation, -5, 5)
                    if evals < budget:
                        cf = func(candidate)
                        evals += 1
                        if cf < gbest_f:
                            gbest_f = cf
                            gbest_x = candidate.copy()

        # Final local refinement with (1+1)-ES
        remaining = budget - evals
        if remaining > 5 and time.time() < deadline - 5:
            x = gbest_x.copy()
            fx = gbest_f
            sigma = 0.05
            no_improve = 0
            limit = 10

            while remaining > 0 and time.time() < deadline - 5:
                if no_improve > limit:
                    sigma = max(0.01, sigma * 0.5)
                    no_improve = 0
                    if sigma < 1e-4:
                        break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                remaining -= 1
                evals += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < gbest_f:
                        gbest_f = fx
                        gbest_x = x.copy()
                    sigma *= 1.05
                    no_improve = 0
                else:
                    sigma *= 0.95
                    no_improve += 1

        return gbest_x, gbest_f
# EVOLVE-BLOCK-END
