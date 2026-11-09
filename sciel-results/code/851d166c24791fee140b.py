import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290  # Safety margin for 300s limit
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(123)
        
        # PSO Parameters
        n_particles = min(20, max(10, budget // 25))
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        
        # Initialize particles with Latin Hypercube Sampling
        positions = rng.uniform(-5, 5, size=(n_particles, dim))
        velocities = rng.uniform(-1, 1, size=(n_particles, dim))
        
        # Evaluate initial positions
        pbest_positions = np.zeros_like(positions)
        pbest_values = np.full(n_particles, float('inf'))
        gbest_position = np.zeros(dim)
        gbest_value = float('inf')
        
        for i in range(n_particles):
            if eval_count >= budget:
                break
            x = np.clip(positions[i], -5.0, 5.0)
            fx = func(x)
            eval_count += 1
            
            if fx < pbest_values[i]:
                pbest_values[i] = fx
                pbest_positions[i] = x.copy()
            
            if fx < gbest_value:
                gbest_value = fx
                gbest_position = x.copy()
        
        if eval_count >= budget:
            return gbest_position, gbest_value
        
        # PSO Main Loop
        while eval_count < budget and time.time() < deadline - 10:
            # Check for time constraint
            if time.time() > deadline - 10:
                break
            
            # Update velocities and positions for each particle
            for i in range(n_particles):
                if eval_count >= budget:
                    break
                
                # Random coefficients
                r1 = rng.random()
                r2 = rng.random()
                
                # Velocity update
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * (pbest_positions[i] - positions[i]) + 
                               c2 * r2 * (gbest_position - positions[i]))
                
                # Position update
                positions[i] = positions[i] + velocities[i]
                
                # Clip to bounds
                positions[i] = np.clip(positions[i], -5.0, 5.0)
                
                # Evaluate new position
                x = positions[i].copy()
                fx = func(x)
                eval_count += 1
                
                # Update personal best
                if fx < pbest_values[i]:
                    pbest_values[i] = fx
                    pbest_positions[i] = x.copy()
                
                # Update global best
                if fx < gbest_value:
                    gbest_value = fx
                    gbest_position = x.copy()
            
            # Adaptive inertia: decrease w over time for convergence
            progress = 1.0 - (eval_count / budget)
            w = 0.9 - 0.4 * (1.0 - progress)
        
        # Phase 3: Local Polish with (1+1)-ES from PSO best
        remaining = budget - eval_count
        if remaining > 0 and time.time() < deadline - 5:
            rng_local = np.random.default_rng(42)
            x = gbest_position.copy()
            fx = gbest_value
            sigma = 0.05
            no_improve = 0
            limit = 15
            
            while remaining > 0 and time.time() < deadline - 5:
                if no_improve > limit:
                    sigma = max(0.01, sigma * 2.0)
                    no_improve = 0
                
                y = np.clip(x + sigma * rng_local.standard_normal(dim), -5, 5)
                fy = func(y)
                remaining -= 1
                eval_count += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < gbest_value:
                        gbest_value = fx
                        gbest_position = x.copy()
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1
                    if sigma < 1e-6:
                        break

        return gbest_position, gbest_value


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
