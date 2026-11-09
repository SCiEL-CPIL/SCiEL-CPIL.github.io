"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-scale population-based optimizer with adaptive diversity and exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Track best solution
        best_x = None
        best_f = np.inf
        eval_count = 0

        def check_time():
            return time.time() < deadline - 3.0

        def try_update(x, fx):
            nonlocal best_x, best_f, eval_count
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        def clip_box(x):
            return np.clip(x, -5.0, 5.0)

        # ===== Phase 1: Diverse initialization =====
        # Start with a diverse set of initial points to cover the search space
        n_init = min(20, budget - 10)
        if n_init < 5:
            n_init = max(3, budget // 10)
        
        # Mix of strategies for initialization
        init_points = []
        # Origin
        init_points.append(np.zeros(n))
        # Random uniform in [-4, 4]
        for _ in range(max(1, n_init - 5)):
            init_points.append(rng.uniform(-4.0, 4.0, n))
        # Random with smaller scale
        for _ in range(max(1, n_init - 10)):
            init_points.append(rng.uniform(-2.0, 2.0, n))
        # Axis-aligned points
        for axis in range(min(n, 3)):
            pt = np.zeros(n)
            pt[axis] = rng.choice([-3.0, 3.0])
            init_points.append(pt)
        
        # Evaluate initial points
        for pt in init_points[:n_init]:
            if not check_time() or eval_count >= budget:
                break
            pt = clip_box(pt)
            fv = func(pt)
            try_update(pt, fv)
        
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            try_update(best_x, best_f)

        # ===== Phase 2: Adaptive (1+λ)-ES with population =====
        # Main optimization phase: evolve a population with adaptive step size
        
        lambda_pop = 8  # Population size for offspring
        sigma = 1.0     # Initial step size
        no_improve = 0
        max_no_improve = 50
        sigma_min = 1e-5
        sigma_max = 5.0
        
        # Maintain a small population of diverse points
        population = [(best_x.copy(), best_f)]
        
        # Add some diversity to population
        for _ in range(min(3, budget - eval_count - 5)):
            if not check_time() or eval_count >= budget:
                break
            x_div = clip_box(best_x + rng.standard_normal(n) * 1.0)
            f_div = func(x_div)
            try_update(x_div, f_div)
            population.append((x_div, f_div))
        
        # Sort population by fitness
        population.sort(key=lambda t: t[1])
        
        while check_time() and eval_count < budget:
            # Generate offspring from multiple parents
            offspring = []
            n_offspring = min(lambda_pop, budget - eval_count)
            
            for _ in range(n_offspring):
                if not check_time() or eval_count >= budget:
                    break
                
                # Select parent: mostly from best, sometimes from others for diversity
                if rng.random() < 0.6 and len(population) > 0:
                    parent = population[0][0]
                elif len(population) > 1:
                    parent = population[rng.integers(0, min(3, len(population)))][0]
                else:
                    parent = best_x
                
                # Generate child with adaptive noise
                z = rng.standard_normal(n)
                child = clip_box(parent + sigma * z)
                f_child = func(child)
                try_update(child, f_child)
                offspring.append((child, f_child))
            
            if not offspring:
                break
            
            # Update population with offspring
            population.extend(offspring)
            population.sort(key=lambda t: t[1])
            
            # Keep only top population_size points
            max_pop_size = min(10, budget - eval_count - lambda_pop)
            if max_pop_size < 1:
                max_pop_size = 1
            population = population[:max_pop_size]
            
            # Check for improvement
            if population[0][1] < best_f - 1e-10:
                best_f = population[0][1]
                best_x = population[0][0].copy()
                no_improve = 0
                # Increase sigma slightly on improvement
                sigma = min(sigma * 1.15, sigma_max)
            else:
                no_improve += 1
                # Decrease sigma on stagnation
                sigma = max(sigma * 0.92, sigma_min)
            
            # Occasionally inject diversity to avoid local optima
            if no_improve > 20 and eval_count < budget - 5:
                if not check_time():
                    break
                x_inject = clip_box(rng.uniform(-4.0, 4.0, n))
                f_inject = func(x_inject)
                try_update(x_inject, f_inject)
                population.append((x_inject, f_inject))
                population.sort(key=lambda t: t[1])
                population = population[:max_pop_size]
                no_improve = 0
                sigma = max(sigma * 0.8, sigma_min)
            
            # If population is very stagnant and sigma is small, try a larger jump
            if no_improve > 40 and sigma < 0.1 and eval_count < budget - 10:
                if not check_time():
                    break
                # Large jump from best
                x_jump = clip_box(best_x + rng.standard_normal(n) * 2.0)
                f_jump = func(x_jump)
                try_update(x_jump, f_jump)
                if f_jump < population[0][1]:
                    population[0] = (x_jump, f_jump)
                    best_f = f_jump
                    best_x = x_jump.copy()
                    no_improve = 0
                    sigma = 1.0
                else:
                    # Try from a random point
                    x_rand = clip_box(rng.uniform(-4.0, 4.0, n))
                    f_rand = func(x_rand)
                    try_update(x_rand, f_rand)
                    if f_rand < population[0][1]:
                        population[0] = (x_rand, f_rand)
                        best_f = f_rand
                        best_x = x_rand.copy()
                        no_improve = 0
                        sigma = 1.0
                    else:
                        no_improve = 0  # Reset to avoid infinite loop
                        sigma = max(sigma * 1.5, 0.5)

        # ===== Phase 3: Fine-tuning with small steps =====
        # Use remaining budget for precise local search
        if check_time() and eval_count < budget:
            sigma_fine = max(sigma * 0.1, 0.01)
            x_fine = best_x.copy()
            f_fine = best_f
            no_improve_fine = 0
            max_fine_steps = min(30, budget - eval_count)
            
            for _ in range(max_fine_steps):
                if not check_time() or eval_count >= budget:
                    break
                
                z = rng.standard_normal(n)
                y = clip_box(x_fine + sigma_fine * z)
                fy = func(y)
                try_update(y, fy)
                
                if fy < f_fine:
                    x_fine = y
                    f_fine = fy
                    no_improve_fine = 0
                    sigma_fine = min(sigma_fine * 1.1, 0.5)
                else:
                    sigma_fine *= 0.9
                    no_improve_fine += 1
                
                if sigma_fine < sigma_min or no_improve_fine > 20:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
