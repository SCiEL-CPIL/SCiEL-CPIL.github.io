"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Derandomized Discrete Particle Swarm Optimization (ddPSO) with ring topology.
    Uses a fixed maximum velocity bound and dynamic inertia weight decreasing
    linearly from 0.9 to 0.4 over the budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        calls_made = 0
        best_x_global = None
        best_f_global = float('inf')

        def safe_call(x):
            nonlocal calls_made, best_x_global, best_f_global
            if calls_made >= budget:
                return None
            try:
                val = float(func(x))
            except Exception:
                return None
            calls_made += 1
            if val < best_f_global:
                best_f_global = val
                best_x_global = x.copy()
            return val

        rng = np.random.default_rng(42)

        # Initialize swarm
        n_particles = 10
        positions = np.empty((n_particles, dim))
        velocities = np.zeros((n_particles, dim))
        personal_bests = np.empty((n_particles, dim))
        personal_best_fits = np.full(n_particles, float('inf'))

        # Latin Hypercube Sampling for initial positions
        for i in range(n_particles):
            if not check_time() or calls_made >= budget:
                break
            x = np.empty(dim)
            for d in range(dim):
                perm = rng.permutation(n_particles)
                x[d] = -5.0 + 10.0 * (perm[i] + rng.random()) / n_particles
            x = np.clip(x, -5, 5)
            fx = safe_call(x)
            if fx is None:
                break
            positions[i] = x
            personal_bests[i] = x
            personal_best_fits[i] = fx

        # Ensure at least one valid particle
        valid_particles = np.where(personal_best_fits < float('inf'))[0]
        if len(valid_particles) == 0:
            return best_x_global, best_f_global

        # Initialize global best from valid particles
        global_idx = valid_particles[np.argmin(personal_best_fits[valid_particles])]
        best_x_global = personal_bests[global_idx].copy()
        best_f_global = personal_best_fits[global_idx]

        # Ring topology neighbors
        # Each particle i talks to i-1 and i+1 (mod n)
        def get_neighbors(i):
            left = (i - 1) % n_particles
            right = (i + 1) % n_particles
            return [left, right]

        # PSO parameters
        c1 = 1.5  # Cognitive
        c2 = 1.5  # Social
        vmax = 1.0  # Max velocity
        w_start = 0.9
        w_end = 0.4

        # Total steps planned for inertia decay
        # We have roughly budget - calls_made left
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x_global, best_f_global

        # We will iterate until budget or time runs out.
        # Inertia decays linearly over the *remaining* budget.
        step = 0

        while check_time() and calls_made < budget:
            step += 1
            # Dynamic inertia weight
            progress = min(1.0, step / remaining_budget)
            w = w_start + (w_end - w_start) * progress

            # Update each particle
            for i in range(n_particles):
                if not check_time() or calls_made >= budget:
                    break

                # Get social best from ring neighbors
                neighbors = get_neighbors(i)
                neighbor_fits = personal_best_fits[neighbors]
                neighbor_idxs = neighbors
                # Pick the best neighbor
                best_neighbor_idx = neighbor_idxs[np.argmin(neighbor_fits)]
                social_best = personal_bests[best_neighbor_idx]

                # Update velocity
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_bests[i] - positions[i]) +
                                 c2 * r2 * (social_best - positions[i]))

                # Clamp velocity
                velocities[i] = np.clip(velocities[i], -vmax, vmax)

                # Update position
                new_pos = positions[i] + velocities[i]
                new_pos = np.clip(new_pos, -5, 5)

                # Evaluate
                fx = safe_call(new_pos)
                if fx is None:
                    break

                positions[i] = new_pos

                # Update personal best
                if fx < personal_best_fits[i]:
                    personal_best_fits[i] = fx
                    personal_bests[i] = new_pos

                    # Update global best
                    if fx < best_f_global:
                        best_f_global = fx
                        best_x_global = new_pos.copy()

                # If we hit the budget limit in the middle of the loop, break out
                if calls_made >= budget:
                    break

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
