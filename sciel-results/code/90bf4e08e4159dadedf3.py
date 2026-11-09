"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Surrogate-guided multi-start optimizer with coordinate line search and adaptive population ES."""

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

        # ===== Phase 1: Structured Initialization =====
        # Evaluate a diverse set of points to find a good starting basin.
        # This is crucial for AOCC as early progress is heavily weighted.
        n_init = min(15, budget - 20)
        init_points = []
        init_points.append(np.zeros(n))
        
        # Random uniform points in the likely region [-4, 4]
        for _ in range(max(1, n_init // 3)):
            init_points.append(rng.uniform(-4.0, 4.0, n))
        
        # Closer to center
        for _ in range(max(1, n_init // 3)):
            init_points.append(rng.uniform(-2.0, 2.0, n))
        
        # Axis-aligned probes (helps if function is separable like Sphere)
        for axis in range(min(n, 3)):
            pt = np.zeros(n)
            pt[axis] = rng.choice([-3.0, 3.0])
            init_points.append(pt)

        for pt in init_points[:n_init]:
            if not check_time() or eval_count >= budget:
                break
            pt = clip_box(pt)
            fv = func(pt)
            try_update(pt, fv)

        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(n)
            best_f = func(best_x)
            try_update(best_x, best_f)

        # ===== Phase 2: Coordinate Line Search from Best =====
        # Exploit the separable structure of BBOB functions (Sphere, Rastrigin, etc.)
        # by doing a 1D search along each coordinate from the current best.
        if check_time() and eval_count < budget:
            # Scan points along each axis relative to current best
            # We check a few offsets from the current best coordinate
            offsets = [2.0, -2.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25]
            
            for coord in range(n):
                if not check_time() or eval_count >= budget:
                    break
                
                current_val = best_x[coord]
                best_coord_val = current_val
                best_coord_f = best_f
                
                for off in offsets:
                    if not check_time() or eval_count >= budget:
                        break
                    
                    new_val = current_val + off
                    # Clip to box
                    new_val = np.clip(new_val, -5.0, 5.0)
                    
                    x_try = best_x.copy()
                    x_try[coord] = new_val
                    
                    f_try = func(x_try)
                    try_update(x_try, f_try)
                    
                    if f_try < best_coord_f:
                        best_coord_f = f_try
                        best_coord_val = new_val
                
                # Update best_x with the best coordinate found
                if best_coord_val != current_val:
                    best_x[coord] = best_coord_val
                    # Note: best_f is already updated by try_update if this was the global best
                    # But we need to ensure best_x is consistent.
                    # If best_coord_f < best_f, best_x was already updated in try_update.
                    # However, if the improvement was local to this coord but not global best,
                    # we should still move towards it? 
                    # Actually, for AOCC, we just need to find the global min.
                    # Moving to a slightly better point that isn't the global best yet
                    # is fine as long as we continue.
                    # Let's just keep the best_x as the global best found so far.
                    # The coordinate search effectively does a local refinement.
                    # If the global best moved, best_x is already correct.
                    # If not, we might want to update best_x to this "locally better" point?
                    # No, best_x is the global best. We should only update best_x if f_try < best_f.
                    # The loop above updates best_x via try_update.
                    # So best_x is always the global best.
                    # We don't need to explicitly set best_x[coord] unless we are sure it's better globally.
                    # But wait, if f_try < best_coord_f (local best) but f_try > best_f (global best),
                    # then best_x remains the global best. This is correct.
                    # The "search" is just evaluating points.
        
        # ===== Phase 3: Adaptive Population ES =====
        # Use a population-based evolution strategy to refine the solution.
        # This helps with non-convex functions like Rastrigin and Gallagher.
        
        lambda_pop = 6
        sigma = 0.5
        no_improve = 0
        max_no_improve = 30
        sigma_min = 1e-6
        sigma_max = 5.0

        # Initialize population with the best point and some diverse points
        population = [(best_x.copy(), best_f)]
        for _ in range(min(4, budget - eval_count - 5)):
            if not check_time() or eval_count >= budget:
                break
            x_div = clip_box(best_x + rng.standard_normal(n) * 1.0)
            f_div = func(x_div)
            try_update(x_div, f_div)
            population.append((x_div, f_div))
        
        population.sort(key=lambda t: t[1])

        while check_time() and eval_count < budget:
            n_offspring = min(lambda_pop, budget - eval_count)
            offspring = []
            
            for _ in range(n_offspring):
                if not check_time() or eval_count >= budget:
                    break
                
                # Selection: Use the best parent with high probability, occasionally use others
                if rng.random() < 0.7 and len(population) > 0:
                    parent = population[0][0]
                elif len(population) > 1:
                    parent = population[rng.integers(0, min(3, len(population)))][0]
                else:
                    parent = best_x
                
                child = clip_box(parent + sigma * rng.standard_normal(n))
                f_child = func(child)
                try_update(child, f_child)
                offspring.append((child, f_child))

            if not offspring:
                break

            # Update population
            population.extend(offspring)
            population.sort(key=lambda t: t[1])
            
            # Keep only the best max_pop_size individuals
            max_pop_size = min(10, budget - eval_count - lambda_pop)
            if max_pop_size < 1:
                max_pop_size = 1
            population = population[:max_pop_size]

            # Check for improvement
            if population[0][1] < best_f - 1e-10:
                best_f = population[0][1]
                best_x = population[0][0].copy()
                no_improve = 0
                sigma = min(sigma * 1.1, sigma_max)
            else:
                no_improve += 1
                sigma = max(sigma * 0.9, sigma_min)

            # Diversity injection on stagnation
            if no_improve > 15 and eval_count < budget - 10:
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

            # Large jump on severe stagnation
            if no_improve > 25 and sigma < 0.1 and eval_count < budget - 15:
                if not check_time():
                    break
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
                    # Reset sigma to encourage exploration
                    no_improve = 0
                    sigma = max(sigma * 1.5, 0.5)

        # ===== Phase 4: Fine-tuning =====
        # If we have budget left, do a final (1+1)-ES refinement with small sigma
        if check_time() and eval_count < budget:
            sigma_fine = max(sigma * 0.1, 0.01)
            x_fine = best_x.copy()
            f_fine = best_f
            no_improve_fine = 0
            max_fine_steps = min(20, budget - eval_count)
            
            for _ in range(max_fine_steps):
                if not check_time() or eval_count >= budget:
                    break
                y = clip_box(x_fine + sigma_fine * rng.standard_normal(n))
                fy = func(y)
                try_update(y, fy)
                
                if fy < f_fine - 1e-12:
                    x_fine = y
                    f_fine = fy
                    no_improve_fine = 0
                    sigma_fine = min(sigma_fine * 1.05, 0.5)
                else:
                    sigma_fine *= 0.9
                    no_improve_fine += 1
                
                if sigma_fine < sigma_min or no_improve_fine > 10:
                    break
            
            if f_fine < best_f:
                best_x = x_fine
                best_f = f_fine

        return best_x, best_f
# EVOLVE-BLOCK-END
