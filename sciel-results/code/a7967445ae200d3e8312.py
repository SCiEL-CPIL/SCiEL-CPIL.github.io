import numpy as np


class Optimizer:
    """Particle Swarm Optimization (PSO) for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.NP = 20  # Swarm size
        self.w_init = 0.9
        self.w_end = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
        self.v_max_factor = 0.2  # Max velocity as fraction of range

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        NP = self.NP
        budget = self.budget
        
        # Bounds
        lower = -5.0
        upper = 5.0
        range_ = upper - lower
        v_max = self.v_max_factor * range_
        
        # Initialize positions and velocities
        pop = rng.uniform(lower, upper, (NP, dim))
        vels = rng.uniform(-v_max, v_max, (NP, dim))
        
        # Evaluate initial population
        fvals = np.array([func(x) for x in pop])
        budget -= NP
        
        # Initialize personal and global bests
        pbest = pop.copy()
        pbest_fvals = fvals.copy()
        
        gbest_idx = np.argmin(fvals)
        gbest = pop[gbest_idx].copy()
        gbest_fval = fvals[gbest_idx]
        
        if budget <= 0:
            return gbest, gbest_fval
        
        # Determine number of iterations we can do
        # Each iteration costs NP evaluations
        # We might not have enough budget for full iterations, so we handle partials
        # But standard PSO updates all particles. 
        # If budget < NP, we can't do a full iteration.
        # We will loop while budget > 0.
        # Inside, we try to update as many particles as possible.
        
        iter_count = 0
        total_iters = budget // NP # Approximate
        
        while budget > 0:
            # Update inertia weight linearly
            if total_iters > 0:
                progress = iter_count / total_iters
            else:
                progress = 0.0
            w = self.w_init + (self.w_end - self.w_init) * progress
            
            for i in range(NP):
                if budget <= 0:
                    break
                
                # Update velocity for particle i
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                
                vels[i] = (w * vels[i] + 
                           self.c1 * r1 * (pbest[i] - pop[i]) + 
                           self.c2 * r2 * (gbest - pop[i]))
                
                # Clamp velocity
                vels[i] = np.clip(vels[i], -v_max, v_max)
                
                # Update position
                pop[i] = pop[i] + vels[i]
                
                # Clip to bounds
                pop[i] = np.clip(pop[i], lower, upper)
                
                # Evaluate
                f_new = func(pop[i])
                budget -= 1
                
                # Update personal best
                if f_new < pbest_fvals[i]:
                    pbest_fvals[i] = f_new
                    pbest[i] = pop[i].copy()
                    
                    # Update global best
                    if f_new < gbest_fval:
                        gbest_fval = f_new
                        gbest = pop[i].copy()
                        
            iter_count += 1

        return gbest, gbest_fval
