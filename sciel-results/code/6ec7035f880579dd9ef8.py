"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Particle Swarm Optimization (PSO) for black-box continuous optimization.
    Uses a swarm of particles to explore the search space.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n_particles = 20
        calls_made = 0
        
        # Initialize particles uniformly in the box [-5, 5]^dim
        positions = rng.uniform(-5, 5, (n_particles, self.dim))
        velocities = rng.uniform(-1, 1, (n_particles, self.dim))
        
        # Track personal and global bests
        pbest_f = np.full(n_particles, np.inf)
        pbest_x = np.zeros((n_particles, self.dim))
        gbest_f = np.inf
        gbest_x = None
        
        # Evaluate initial population
        for i in range(n_particles):
            if calls_made >= self.budget: 
                break
            f = func(positions[i])
            calls_made += 1
            pbest_f[i] = f
            pbest_x[i] = positions[i].copy()
            if f < gbest_f:
                gbest_f = f
                gbest_x = positions[i].copy()

        # PSO Parameters
        w_start = 0.9
        w_end = 0.4
        c1 = 1.49445
        c2 = 1.49445
        
        # Calculate max iterations allowed by budget
        # Each iteration evaluates n_particles points
        max_iter = max(1, (self.budget - calls_made) // n_particles)
        
        for t in range(max_iter):
            if calls_made >= self.budget:
                break
                
            # Linearly decrease inertia weight
            w = w_start + (w_end - w_start) * (t / max_iter)
            
            # Update velocities
            r1 = rng.random((n_particles, 1))
            r2 = rng.random((n_particles, 1))
            
            v_new = w * velocities + c1 * r1 * (pbest_x - positions) + c2 * r2 * (gbest_x - positions)
            
            # Clip velocities to prevent divergence
            v_new = np.clip(v_new, -1.0, 1.0)
            velocities = v_new
            
            # Update positions
            positions = positions + velocities
            positions = np.clip(positions, -5, 5)
            
            # Evaluate new positions
            for i in range(n_particles):
                if calls_made >= self.budget:
                    break
                f = func(positions[i])
                calls_made += 1
                
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest_x[i] = positions[i].copy()
                    if f < gbest_f:
                        gbest_f = f
                        gbest_x = positions[i].copy()

        # Fallback if no calls were made (should not happen)
        if gbest_x is None:
             gbest_x = np.zeros(self.dim)
             gbest_f = func(gbest_x)
             
        return gbest_x, gbest_f
# EVOLVE-BLOCK-END
