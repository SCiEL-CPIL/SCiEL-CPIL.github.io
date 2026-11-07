"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Particle Swarm Optimization (PSO) for continuous black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # PSO Parameters
        n_particles = 20
        # Calculate how many full generations we can afford
        # We need at least 1 call per particle per generation
        max_generations = (budget - n_particles) // n_particles
        if max_generations < 1:
            max_generations = 1

        # Initialize Particles
        particles = rng.uniform(-5, 5, (n_particles, n))
        velocities = rng.uniform(-1, 1, (n_particles, n))
        
        pbest = np.zeros_like(particles)
        pbest_f = np.full(n_particles, np.inf)
        gbest = np.zeros(n)
        gbest_f = np.inf
        
        calls_used = 0

        # Initial Evaluation
        for i in range(n_particles):
            if calls_used >= budget:
                break
            f = func(particles[i])
            calls_used += 1
            pbest[i] = particles[i].copy()
            pbest_f[i] = f
            if f < gbest_f:
                gbest_f = f
                gbest = particles[i].copy()

        best_x = gbest.copy()
        best_f = gbest_f

        # PSO Weights
        w = 0.7  # Inertia
        c1 = 1.49445  # Cognitive
        c2 = 1.49445  # Social
        
        # Main Loop
        for gen in range(max_generations):
            for i in range(n_particles):
                if calls_used >= budget:
                    break
                
                r1 = rng.random(n)
                r2 = rng.random(n)
                
                # Update Velocity
                velocities[i] = (w * velocities[i] + 
                                 c1 * r1 * (pbest[i] - particles[i]) + 
                                 c2 * r2 * (gbest - particles[i]))
                
                # Limit Velocity
                velocities[i] = np.clip(velocities[i], -2.0, 2.0)
                
                # Update Position
                particles[i] = particles[i] + velocities[i]
                
                # Enforce Bounds
                particles[i] = np.clip(particles[i], -5, 5)
                
                # Evaluate
                f = func(particles[i])
                calls_used += 1
                
                # Update Personal Best
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest[i] = particles[i].copy()
                    
                # Update Global Best
                if f < gbest_f:
                    gbest_f = f
                    gbest = particles[i].copy()
                    best_f = f
                    best_x = gbest.copy()

            # Decay Inertia to help convergence in later stages
            w = max(0.4, w * 0.95)

        return best_x, best_f
# EVOLVE-BLOCK-END
