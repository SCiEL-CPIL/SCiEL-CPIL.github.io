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
        lam = 12
        sigma = 2.0
        
        stagnation_count = 0
        restart_threshold = 100
        last_improvement = 0
        
        while budget > 0 and time.time() < deadline:
            # Generate offspring
            offspring = []
            for _ in range(lam):
                if time.time() > deadline or budget <= 0:
                    break
                # Use best_x as center with some diversity
                if rng.random() < 0.15:
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
                last_improvement = budget
            else:
                stagnation_count += 1
            
            # Adapt sigma based on success rate
            improved = 0
            for _, fc in combined[1:]:
                if fc < best_f:
                    improved += 1
            success_rate = improved / max(1, len(combined) - 1)
            
            # 1/5 success rule with smoothing
            if success_rate > 0.3:
                sigma *= 1.05
            elif success_rate < 0.1:
                sigma *= 0.95
            sigma = np.clip(sigma, 0.005, 5.0)
            
            # If no improvement for a long time, try a larger jump
            if stagnation_count > 50:
                sigma = max(sigma, 1.0)
                # Try a random restart from a different region
                if rng.random() < 0.5:
                    new_x = rng.uniform(-4, 4, dim)
                    if budget > 0 and time.time() < deadline:
                        new_f = func(new_x)
                        budget -= 1
                        if new_f < best_f:
                            best_x = new_x
                            best_f = new_f
                            stagnation_count = 0
                            last_improvement = budget
                stagnation_count = 0
        
        # Final polish: if we have budget left, do local refinement around best_x
        if budget > 10 and time.time() < deadline:
            polish_sigma = max(0.01, sigma * 0.1)
            for _ in range(min(budget, 20)):
                if time.time() > deadline or budget <= 0:
                    break
                child = np.clip(best_x + polish_sigma * rng.standard_normal(dim), -5, 5)
                fc = func(child)
                budget -= 1
                if fc < best_f:
                    best_f = fc
                    best_x = child
                    polish_sigma *= 0.8
        
        return best_x, best_f
# EVOLVE-BLOCK-END
