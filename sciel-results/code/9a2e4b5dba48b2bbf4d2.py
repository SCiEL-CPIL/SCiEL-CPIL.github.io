"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Particle Swarm Optimization (PSO) continuous optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # PSO Parameters
        n_particles = 20
        w_start = 0.9
        w_end = 0.4
        c1 = 1.5
        c2 = 1.5
        v_max_factor = 0.5  # Max velocity as fraction of domain range

        # Domain
        lower = -5.0
        upper = 5.0
        range_val = upper - lower
        v_max = v_max_factor * range_val

        # Initialize particles
        # Positions
        positions = rng.uniform(lower, upper, (n_particles, n))
        # Velocities
        velocities = rng.uniform(-v_max, v_max, (n_particles, n))

        # Initialize bests
        p_best_pos = positions.copy()
        p_best_val = np.full(n_particles, np.inf)
        g_best_pos = np.zeros(n)
        g_best_val = np.inf

        calls_made = 0

        # Initial evaluation
        for i in range(n_particles):
            f_val = func(positions[i])
            calls_made += 1
            if f_val < p_best_val[i]:
                p_best_val[i] = f_val
                p_best_pos[i] = positions[i]
            if f_val < g_best_val:
                g_best_val = f_val
                g_best_pos = positions[i]

        if calls_made >= budget:
            return g_best_pos, g_best_val

        # Total iterations we can do
        # Each iteration consumes n_particles calls
        # We can do floor((budget - calls_made) / n_particles) full iterations
        # And possibly some partial ones? No, PSO works in batches.
        # We will loop until we run out of budget.
        
        total_iters_possible = (budget - calls_made) // n_particles
        
        for t in range(total_iters_possible):
            # Update inertia weight
            if total_iters_possible > 0:
                progress = t / max(1, total_iters_possible - 1)
            else:
                progress = 0
            w = w_start + (w_end - w_start) * progress

            for i in range(n_particles):
                # Update velocity
                r1 = rng.random(n)
                r2 = rng.random(n)
                
                # v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                velocities[i] = (w * velocities[i] + 
                                c1 * r1 * (p_best_pos[i] - positions[i]) + 
                                c2 * r2 * (g_best_pos - positions[i]))
                
                # Clip velocity
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                
                # Update position
                positions[i] = positions[i] + velocities[i]
                
                # Clip position to domain
                positions[i] = np.clip(positions[i], lower, upper)
                
                # Evaluate
                f_val = func(positions[i])
                calls_made += 1
                
                # Update pbest
                if f_val < p_best_val[i]:
                    p_best_val[i] = f_val
                    p_best_pos[i] = positions[i]
                
                # Update gbest
                if f_val < g_best_val:
                    g_best_val = f_val
                    g_best_pos = positions[i]
                    
                # If we hit the budget exactly, we can stop early
                if calls_made >= budget:
                    break
                    
            if calls_made >= budget:
                break

        return g_best_pos, g_best_val
# EVOLVE-BLOCK-END
