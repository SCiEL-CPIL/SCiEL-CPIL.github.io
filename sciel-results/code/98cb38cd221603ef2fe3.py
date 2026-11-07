import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # PSO (Particle Swarm Optimization) implementation
        # Structural change from CMA-ES to PSO
        
        # Swarm size: 20 particles
        # 500 budget / 20 = 25 generations
        swarm_size = 20
        
        # Initialize positions uniformly in [-5, 5]
        pos = np.random.uniform(-5, 5, (swarm_size, n))
        # Initialize velocities uniformly in [-1, 1]
        vel = np.random.uniform(-1, 1, (swarm_size, n))
        
        # Evaluate initial swarm
        pbest_pos = pos.copy()
        pbest_val = np.array([func(x) for x in pos])
        calls = swarm_size
        
        gbest_idx = np.argmin(pbest_val)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_val = pbest_val[gbest_idx]
        
        # PSO parameters
        w = 0.7  # Inertia weight
        c1 = 1.5 # Cognitive coefficient
        c2 = 1.5 # Social coefficient
        
        t_start = time.time()
        deadline = t_start + 290.0
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Update each particle
            for i in range(swarm_size):
                r1 = np.random.rand(n)
                r2 = np.random.rand(n)
                
                # Update velocity
                vel[i] = w * vel[i] + c1 * r1 * (pbest_pos[i] - pos[i]) + c2 * r2 * (gbest_pos - pos[i])
                
                # Clip velocity to avoid large jumps
                vel[i] = np.clip(vel[i], -1.0, 1.0)
                
                # Update position
                pos[i] = pos[i] + vel[i]
                
                # Clip position to box [-5, 5]
                pos[i] = np.clip(pos[i], -5, 5)
                
                # Evaluate
                val = func(pos[i])
                calls += 1
                
                # Update personal best
                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest_pos[i] = pos[i].copy()
                    
                    # Update global best
                    if val < gbest_val:
                        gbest_val = val
                        gbest_pos = pos[i].copy()
            
            # Decay inertia weight to improve convergence
            w = max(0.4, w * 0.95)

        return gbest_pos, gbest_val
