"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB-style Schwefel f20 on [-5, 5]^5.
    
    Strategy:
      - Start with a set of diverse initial points.
      - Use a (mu+lambda)-ES with adaptive step size.
      - Track best point and use it for restarts if stagnation occurs.
      - Deadline guard to ensure we return within 330s wall-clock.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin
        
        rng = np.random.default_rng(42)
        
        # Initial points: origin, random, and a few structured points
        init_points = [np.zeros(dim)]
        for _ in range(4):
            init_points.append(rng.uniform(-5, 5, dim))
        # Add points at boundaries
        init_points.append(np.full(dim, 4.0))
        init_points.append(np.full(dim, -4.0))
        
        # Evaluate initial points
        best_x = None
        best_f = np.inf
        for x0 in init_points:
            if time.time() > deadline:
                break
            x0 = np.clip(x0, -5, 5)
            fx = func(x0)
            budget -= 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            budget -= 1
        
        # (mu+lambda)-ES
        mu = 4
        lam = 12
        sigma = 2.0
        
        # Population
        population = [best_x.copy() + sigma * rng.standard_normal(dim) for _ in range(lam)]
        
        stagnation_count = 0
        restart_threshold = 100
        
        while budget > 0 and time.time() < deadline:
            # Generate offspring
            offspring = []
            for _ in range(lam):
                if time.time() > deadline or budget <= 0:
                    break
                # Use best_x as center with some diversity
                if rng.random() < 0.3:
                    # Occasionally restart from a random point
                    parent = rng.uniform(-5, 5, dim)
                else:
                    parent = best_x
                child = np.clip(parent + sigma * rng.standard_normal(dim), -5, 5)
                offspring.append(child)
            
            if not offspring:
                break
            
            # Evaluate offspring
            offspring_f = []
            for child in offspring:
                if time.time() > deadline or budget <= 0:
                    break
                fc = func(child)
                budget -= 1
                offspring_f.append(fc)
            
            if not offspring_f:
                break
            
            # Combine parent and offspring
            combined = [(best_x, best_f)] + list(zip(offspring, offspring_f))
            combined.sort(key=lambda t: t[1])
            
            new_best_x, new_best_f = combined[0]
            
            if new_best_f < best_f:
                best_x = new_best_x.copy()
                best_f = new_best_f
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Adapt sigma
            if stagnation_count > restart_threshold:
                sigma *= 0.5
                if sigma < 0.01:
                    sigma = 0.01
                stagnation_count = 0
            
            # Step size adaptation based on success rate
            improved = 0
            for _, fc in combined[1:]:
                if fc < best_f:
                    improved += 1
            success_rate = improved / max(1, len(combined) - 1)
            if success_rate > 0.3:
                sigma *= 1.1
            elif success_rate < 0.1:
                sigma *= 0.9
            sigma = np.clip(sigma, 0.01, 5.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
