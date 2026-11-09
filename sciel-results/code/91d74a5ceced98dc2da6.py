"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Particle Swarm Optimization (PSO) followed by Powell's Method local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Allocate budget: 80% to PSO, 20% to Powell
        pso_budget = int(budget * 0.80)
        powell_budget = budget - pso_budget
        
        # PSO Parameters
        N_particles = 20
        # Number of iterations we can afford
        max_iter = max(1, pso_budget // N_particles)
        
        rng = np.random.default_rng(42)
        
        # Initialize positions in [-5, 5]
        positions = rng.uniform(-5, 5, (N_particles, dim))
        # Initialize velocities to zero or small random
        velocities = rng.uniform(-0.5, 0.5, (N_particles, dim))
        
        # Personal bests
        pbest_positions = positions.copy()
        pbest_fvals = np.array([func(np.clip(p, -5, 5)) for p in positions])
        calls = N_particles
        
        # Global best
        global_best_idx = np.argmin(pbest_fvals)
        gbest_position = pbest_positions[global_best_idx].copy()
        gbest_fval = pbest_fvals[global_best_idx]
        
        # PSO Parameters
        w = 0.9  # Inertia weight
        c1 = 1.49445  # Cognitive coefficient
        c2 = 1.49445  # Social coefficient
        
        for t in range(max_iter):
            # Update inertia weight (linear decay)
            w = 0.9 - (0.9 - 0.4) * (t / max_iter)
            
            for i in range(N_particles):
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                
                # Update velocity
                velocities[i] = w * velocities[i] + \
                                c1 * r1 * (pbest_positions[i] - positions[i]) + \
                                c2 * r2 * (gbest_position - positions[i])
                
                # Update position
                positions[i] = positions[i] + velocities[i]
                
                # Clip to bounds
                positions[i] = np.clip(positions[i], -5, 5)
                
                # Evaluate fitness
                if calls >= pso_budget:
                    break
                f_val = func(positions[i])
                calls += 1
                
                # Update personal best
                if f_val < pbest_fvals[i]:
                    pbest_fvals[i] = f_val
                    pbest_positions[i] = positions[i].copy()
                    
                    # Update global best
                    if f_val < gbest_fval:
                        gbest_fval = f_val
                        gbest_position = positions[i].copy()
                        
            if calls >= pso_budget:
                break
        
        # --- Phase 2: Powell's Method ---
        if powell_budget > 0 and calls < budget:
            # Use remaining budget for local refinement
            powell_calls = [0]
            def powell_func(x):
                if powell_calls[0] >= powell_budget:
                    return float('inf')
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                powell_calls[0] += 1
                return val
            
            # Start from best found by PSO
            x0 = gbest_position
            res = minimize(powell_func, x0, method='Powell', 
                           options={'maxiter': powell_budget, 'xatol': 1e-8, 'fatol': 1e-12})
            
            if res.fun < gbest_fval:
                gbest_fval = res.fun
                gbest_position = np.clip(res.x, -5, 5)
        
        return gbest_position, gbest_fval
# EVOLVE-BLOCK-END
