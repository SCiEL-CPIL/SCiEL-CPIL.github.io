"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Particle Swarm Optimization (PSO) for BBOB f3."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # PSO Parameters
        n_particles = 15
        
        # Velocity parameters
        w = 0.7  # Inertia
        c1 = 1.5 # Cognitive
        c2 = 1.5 # Social
        
        # Initialize positions uniformly in [-5, 5]
        positions = rng.uniform(-5, 5, (n_particles, dim))
        
        # Evaluate initial positions
        velocities = np.zeros((n_particles, dim))
        pbests = np.zeros((n_particles, dim))
        pbest_vals = np.full(n_particles, np.inf)
        gbest = None
        gbest_val = np.inf
        
        calls_used = 0
        remaining = budget

        for i in range(n_particles):
            if remaining <= 0:
                break
            x = positions[i]
            fx = func(x)
            calls_used += 1
            remaining -= 1
            
            pbests[i] = x
            pbest_vals[i] = fx
            if fx < gbest_val:
                gbest_val = fx
                gbest = x.copy()

        if remaining <= 0:
            return gbest, gbest_val

        # Main loop
        while remaining >= n_particles:
            # Generate random coefficients
            r1 = rng.random(n_particles)
            r2 = rng.random(n_particles)
            
            # Update velocities
            # v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
            velocities = (w * velocities + 
                          c1 * r1[:, np.newaxis] * (pbests - positions) + 
                          c2 * r2[:, np.newaxis] * (gbest - positions))
            
            # Update positions
            positions = positions + velocities
            
            # Clip positions to [-5, 5]
            positions = np.clip(positions, -5, 5)
            
            # Evaluate new positions
            for i in range(n_particles):
                if remaining <= 0:
                    break
                x = positions[i]
                fx = func(x)
                calls_used += 1
                remaining -= 1
                
                if fx < pbest_vals[i]:
                    pbest_vals[i] = fx
                    pbests[i] = x
                    if fx < gbest_val:
                        gbest_val = fx
                        gbest = x.copy()
        
        # Use remaining budget for local search around gbest if any calls left
        if remaining > 0 and gbest is not None:
            # Simple local search: random perturbations
            step_size = 0.5
            for _ in range(remaining):
                # Decrease step size as we run out of budget? 
                # Or just keep it small.
                x = np.clip(gbest + step_size * rng.standard_normal(dim), -5, 5)
                fx = func(x)
                if fx < gbest_val:
                    gbest_val = fx
                    gbest = x.copy()
                    # Optional: shrink step size on success
                    step_size *= 0.9
                else:
                    step_size *= 0.95
                step_size = max(step_size, 0.01)

        return gbest, gbest_val
# EVOLVE-BLOCK-END
