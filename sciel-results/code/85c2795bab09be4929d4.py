"""Optimized search for BBOB f13 Sharp Ridge."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid optimizer combining:
    1. Random restarts to find the ridge
    2. Population-based ridge following with adaptive step sizes
    3. Nelder-Mead for local refinement
    4. High-precision final polish
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin before 300s limit
        
        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)
        
        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                f = func(x)
            except:
                return None
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global exploration with random sampling
        # Use ~12% of budget for global search
        n_random = max(25, min(60, budget // 8))
        
        for i in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            safe_call(x)
        
        if best_x is None:
            x = np.zeros(dim)
            safe_call(x)
            return best_x, best_f

        # Phase 2: Population-based ridge following
        # Maintain a population of candidates spread along the ridge
        pop_size = 12
        population = [best_x.copy()]
        pop_f = [best_f]
        
        # Initialize population with diverse points near best
        for i in range(1, pop_size):
            if calls >= budget or time.time() > deadline:
                break
            # Create diverse directions
            direction = rng.normal(0, 1, dim)
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            # Vary the distance
            dist = rng.uniform(0.3, 1.5)
            x_try = best_x + dist * direction
            x_try = np.clip(x_try, -5, 5)
            f_try = safe_call(x_try)
            if f_try is not None:
                population.append(x_try)
                pop_f.append(f_try)
        
        # Ridge-following iteration
        ridge_iters = 0
        max_ridge_iters = 200
        stagnant_count = 0
        best_f_ridge = best_f
        
        while calls < budget and ridge_iters < max_ridge_iters and time.time() < deadline:
            ridge_iters += 1
            
            # Sort population by fitness
            idx = np.argsort(pop_f)
            population = [population[i] for i in idx]
            pop_f = [pop_f[i] for i in idx]
            
            # Check stagnation
            if pop_f[0] >= best_f_ridge - 1e-10:
                stagnant_count += 1
            else:
                best_f_ridge = pop_f[0]
                stagnant_count = 0
            
            # If stagnant, try larger perturbations
            if stagnant_count > 15:
                for _ in range(3):
                    if calls >= budget or time.time() > deadline:
                        break
                    direction = rng.normal(0, 1, dim)
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction /= norm
                    for dist in [1.0, 2.0, 3.0]:
                        if calls >= budget or time.time() > deadline:
                            break
                        x_try = best_x + dist * direction
                        x_try = np.clip(x_try, -5, 5)
                        f_try = safe_call(x_try)
                        if f_try is not None and f_try < pop_f[-1]:
                            # Replace worst member
                            population[-1] = x_try
                            pop_f[-1] = f_try
                
                stagnant_count = 0
                continue
            
            # For each member, try to move along the ridge
            for i in range(len(population)):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Estimate gradient via central differences (use 2 calls per dimension)
                # But to save budget, only do this for top few members
                if i < 3:
                    # Try adaptive step in multiple directions
                    n_dirs = 4
                    for d in range(n_dirs):
                        if calls >= budget or time.time() > deadline:
                            break
                        direction = rng.normal(0, 1, dim)
                        norm = np.linalg.norm(direction)
                        if norm > 1e-10:
                            direction /= norm
                        else:
                            direction = np.zeros(dim)
                            direction[d % dim] = 1.0
                        
                        # Adaptive step size based on current spread
                        base_step = 0.2 * (1.0 + 0.5 * np.exp(-ridge_iters / 30.0))
                        step = base_step * rng.uniform(0.5, 2.0)
                        
                        x_try = population[i] + step * direction
                        x_try = np.clip(x_try, -5, 5)
                        f_try = safe_call(x_try)
                        
                        if f_try is None:
                            break
                        
                        if f_try < pop_f[i]:
                            # Accept the move
                            population[i] = x_try
                            pop_f[i] = f_try
                        elif f_try < pop_f[-1]:
                            # Replace worst if not as good as current
                            population[-1] = x_try
                            pop_f[-1] = f_try
            
            # Ensure diversity: if population is too clustered, spread it out
            if len(population) > 2:
                centroid = np.mean(population, axis=0)
                spread = np.std(population, axis=0)
                if np.max(spread) < 0.1:
                    # Add some diversity
                    for i in range(1, min(4, len(population))):
                        if calls >= budget or time.time() > deadline:
                            break
                        direction = rng.normal(0, 1, dim)
                        norm = np.linalg.norm(direction)
                        if norm > 1e-10:
                            direction /= norm
                        dist = rng.uniform(0.3, 0.8)
                        x_try = population[i] + dist * direction
                        x_try = np.clip(x_try, -5, 5)
                        f_try = safe_call(x_try)
                        if f_try is not None:
                            population[i] = x_try
                            pop_f[i] = f_try
        
        # Phase 3: Nelder-Mead refinement on the best point
        if calls < budget and time.time() < deadline:
            center = best_x.copy()
            simplex = [center.copy()]
            f_values = [best_f]
            
            # Smaller initial step for NM since we're already close
            step = 0.3
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                xi = center.copy()
                xi[i] += step
                fi = safe_call(xi)
                if fi is None:
                    break
                simplex.append(xi)
                f_values.append(fi)
            
            if len(simplex) >= dim + 1:
                simplex = np.array(simplex)
                f_values = np.array(f_values)
                
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5
                
                nm_iters = 0
                max_nm_iters = 150
                stagnant_nm = 0
                best_f_nm = best_f
                
                while calls < budget and nm_iters < max_nm_iters and time.time() < deadline:
                    nm_iters += 1
                    
                    idx = np.argsort(f_values)
                    simplex = simplex[idx]
                    f_values = f_values[idx]
                    
                    spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                    f_spread = np.max(f_values) - np.min(f_values)
                    
                    if np.max(spread) < 1e-9 or f_spread < 1e-13:
                        break
                    
                    if f_values[0] >= best_f_nm - 1e-11:
                        stagnant_nm += 1
                    else:
                        best_f_nm = f_values[0]
                        stagnant_nm = 0
                    
                    if stagnant_nm > 15:
                        break
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    fr = safe_call(xr)
                    if fr is None:
                        break
                    
                    if fr < f_values[0]:
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        fe = safe_call(xe)
                        if fe is None:
                            break
                        if fe < fr:
                            simplex[-1] = xe
                            f_values[-1] = fe
                        else:
                            simplex[-1] = xr
                            f_values[-1] = fr
                    elif fr < f_values[-2]:
                        simplex[-1] = xr
                        f_values[-1] = fr
                    else:
                        if fr < f_values[-1]:
                            xc = centroid + rho * (xr - centroid)
                        else:
                            xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -5, 5)
                        fc = safe_call(xc)
                        if fc is None:
                            break
                        if fc <= min(fr, f_values[-1]):
                            simplex[-1] = xc
                            f_values[-1] = fc
                        else:
                            for i in range(1, len(simplex)):
                                if calls >= budget or time.time() > deadline:
                                    break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fi = safe_call(simplex[i])
                                if fi is None:
                                    break
                                f_values[i] = fi
                            if fi is None:
                                break
        
        # Phase 4: High-precision final polish
        # Use remaining budget for very fine-tuned local search
        if calls < budget and time.time() < deadline:
            # Adaptive fine search with shrinking step sizes
            n_polish = min(80, budget - calls)
            for i in range(n_polish):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Step size decreases over time
                progress = i / max(1, n_polish)
                scale = 0.15 * (1.0 - 0.8 * progress) + 0.01
                
                # Try multiple directions
                for d in range(3):
                    if calls >= budget or time.time() > deadline:
                        break
                    direction = rng.normal(0, 1, dim)
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction /= norm
                    else:
                        continue
                    
                    x_try = best_x + scale * direction
                    x_try = np.clip(x_try, -5, 5)
                    safe_call(x_try)
                
                # Also try axis-aligned fine steps
                for axis in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    for delta in [-scale, scale]:
                        if calls >= budget or time.time() > deadline:
                            break
                        x_try = best_x.copy()
                        x_try[axis] += delta
                        x_try = np.clip(x_try, -5, 5)
                        safe_call(x_try)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
