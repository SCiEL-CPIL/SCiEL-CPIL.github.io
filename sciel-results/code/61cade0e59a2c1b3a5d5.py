import numpy as np

class Optimizer:
    """
    Particle Swarm Optimization (PSO) Optimizer.
    Uses a population of particles to explore the search space globally,
    distinct from the local Nelder-Mead approach.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # PSO Parameters
        n_particles = 20
        w = 0.5  # Inertia weight
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter
        
        rng = np.random.RandomState(42)
        
        # Initialize particles
        positions = rng.uniform(-5, 5, size=(n_particles, d))
        velocities = rng.uniform(-1, 1, size=(n_particles, d))
        
        # Evaluate initial positions
        personal_bests = np.zeros_like(positions)
        personal_best_scores = np.full(n_particles, np.inf)
        
        for i in range(n_particles):
            pos = np.clip(positions[i], -5, 5)
            score = func(pos)
            personal_bests[i] = pos
            personal_best_scores[i] = score
        
        global_best_idx = np.argmin(personal_best_scores)
        global_best = personal_bests[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        evals = n_particles
        
        # PSO Loop
        while evals < budget:
            # Determine how many particles we can update in this iteration
            # We update all particles at once, so we need at least n_particles evals left
            if evals + n_particles > budget:
                # If we can't update all, we might do a few random restarts or stop
                # For simplicity, break if we can't do a full iteration, or do partial
                # Let's just break to avoid complex partial updates
                break
                
            # Update velocities and positions
            for i in range(n_particles):
                r1 = rng.rand(d)
                r2 = rng.rand(d)
                
                # Velocity update
                velocities[i] = (w * velocities[i] + 
                                 c1 * r1 * (personal_bests[i] - positions[i]) + 
                                 c2 * r2 * (global_best - positions[i]))
                
                # Position update
                positions[i] = positions[i] + velocities[i]
                
                # Clip to bounds
                positions[i] = np.clip(positions[i], -5, 5)
                
                # Evaluate new position
                score = func(positions[i])
                evals += 1
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_bests[i] = positions[i]
                    
                    # Update global best
                    if score < global_best_score:
                        global_best_score = score
                        global_best = positions[i].copy()
            
            # Optional: Stagnation check / restart if no improvement for many iterations
            # Not strictly necessary for this budget, but good practice.
            
        return global_best, global_best_score
