import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 290
        
        rng = np.random.default_rng(42)
        
        # Phase 1: PSO for global exploration
        # Use smaller population for faster convergence
        n_particles = 10
        pso_budget = int(budget * 0.4)  # 40% of budget for PSO
        if pso_budget < 10:
            pso_budget = budget
            
        max_iters = max(1, pso_budget // n_particles)
        
        # Initialize particles
        positions = rng.uniform(-5, 5, (n_particles, dim))
        velocities = rng.uniform(-1, 1, (n_particles, dim))
        
        pbests = np.copy(positions)
        pbest_fits = np.empty(n_particles)
        
        gbest = positions[0].copy()
        gbest_fit = float('inf')
        
        evals_used = 0
        
        # Evaluate initial positions
        for i in range(n_particles):
            if evals_used >= budget or time.time() > deadline:
                break
            f_val = func(positions[i])
            evals_used += 1
            pbest_fits[i] = f_val
            if f_val < gbest_fit:
                gbest_fit = f_val
                gbest = positions[i].copy()
                
        # PSO iteration
        w_start = 0.9
        w_end = 0.4
        c1 = 1.5
        c2 = 1.5
        
        for t in range(max_iters):
            if evals_used >= budget or time.time() > deadline:
                break
                
            w = w_start - (w_start - w_end) * (t / max_iters)
            
            for i in range(n_particles):
                if evals_used >= budget or time.time() > deadline:
                    break
                    
                r1 = rng.random()
                r2 = rng.random()
                
                velocities[i] = (w * velocities[i] + 
                                 c1 * r1 * (pbests[i] - positions[i]) + 
                                 c2 * r2 * (gbest - positions[i]))
                                 
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], -5, 5)
                
                f_val = func(positions[i])
                evals_used += 1
                
                if f_val < pbest_fits[i]:
                    pbest_fits[i] = f_val
                    pbests[i] = positions[i].copy()
                    
                if f_val < gbest_fit:
                    gbest_fit = f_val
                    gbest = positions[i].copy()
                    
        # Phase 2: (1+1)-ES for local refinement
        # Use remaining budget for aggressive local search
        x = gbest.copy()
        fx = gbest_fit
        sigma = 1.0  # Start with moderate step size
        
        es_budget = budget - evals_used
        if es_budget > 0:
            for _ in range(es_budget):
                if time.time() > deadline:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                evals_used += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.5  # Increase step on success
                else:
                    sigma *= 0.87  # Decrease step on failure
                    
                if evals_used >= budget:
                    break
        
        return x, fx
