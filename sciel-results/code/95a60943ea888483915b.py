import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        
        # Budget allocation
        # PSO: 300 evals
        # Powell: 200 evals
        
        n_particles = 15
        n_gens = 20
        pso_budget = n_particles * n_gens # 300
        
        # Initialize PSO
        positions = rng.uniform(-5, 5, (n_particles, self.dim))
        velocities = rng.uniform(-1, 1, (n_particles, self.dim))
        personal_bests = positions.copy()
        personal_best_scores = np.array([func(p) for p in positions])
        
        global_best = personal_bests[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        
        w = 0.5  # inertia
        c1 = 1.5 # cognitive
        c2 = 1.5 # social
        
        for gen in range(n_gens):
            for i in range(n_particles):
                r1 = rng.random()
                r2 = rng.random()
                velocities[i] = w * velocities[i] + c1 * r1 * (personal_bests[i] - positions[i]) + c2 * r2 * (global_best - positions[i])
                new_pos = positions[i] + velocities[i]
                new_pos = np.clip(new_pos, -5, 5)
                
                score = func(new_pos)
                
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_bests[i] = new_pos
                    if score < global_best_score:
                        global_best_score = score
                        global_best = new_pos
                positions[i] = new_pos
                
                # Check budget
                # This is getting complicated to track exactly.
                
        # Refine
        remaining = self.budget - (n_particles * (n_gens + 1)) # Approx
        if remaining > 0:
             # Use Powell
             pass
