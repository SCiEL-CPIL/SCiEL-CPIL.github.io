"""Multi-Island (1+lambda) Evolution Strategy for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-Island (1+lambda) Evolution Strategy.
    
    Strategy:
    - Maintain N islands (independent populations).
    - Each island runs a (1+lambda)-ES with its own covariance adaptation (or simple step size adaptation).
    - Periodically, the best individual from the best island is copied to other islands to share information,
      while maintaining diversity in the others.
    - This is structurally different from the single-trajectory coordinate descent + local ES.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Global best tracking
        best_x_global = None
        best_f_global = np.inf
        calls_used = 0
        
        # Configuration
        n_islands = 5
        lambda_ = 5  # Offspring count per island per generation
        # Total evaluations per generation = n_islands * lambda_
        # We have 500 budget. 5*5=25 evals/gen. ~20 generations.
        
        # Initialize islands
        # Start with a spread of initial points
        islands = []
        for i in range(n_islands):
            # Initial sigma: large to explore
            sigma = 1.5
            # Initial center: random in [-4, 4]
            x = rng.uniform(-4.0, 4.0, d)
            fx = func(x)
            calls_used += 1
            islands.append({
                'x': x,
                'f': fx,
                'sigma': sigma
            })
            if fx < best_f_global:
                best_f_global = fx
                best_x_global = x.copy()
        
        # Main loop
        # We will iterate until budget or time is up.
        # Each iteration is a "generation" for all islands.
        
        while calls_used < budget:
            if time.time() > deadline:
                break
                
            # Check if we can afford a full generation
            # If remaining budget < n_islands * lambda_, we might do a partial step or stop
            # For simplicity, if we can't afford a full gen, we do single-step refinements on the best island
            if budget - calls_used < n_islands * lambda_:
                # Fallback: Single step (1+1)-ES on the best island to use remaining budget
                best_island = min(islands, key=lambda i: i['f'])
                while calls_used < budget:
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(d)
                    y = best_island['x'] + best_island['sigma'] * z
                    y = np.clip(y, -5.0, 5.0)
                    fy = func(y)
                    calls_used += 1
                    if fy < best_island['f']:
                        best_island['x'] = y
                        best_island['f'] = fy
                        best_island['sigma'] *= 1.1
                    else:
                        best_island['sigma'] *= 0.9
                    
                    if fy < best_f_global:
                        best_f_global = fy
                        best_x_global = y.copy()
                break
            
            # Perform one generation for each island
            for island in islands:
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Generate lambda_ offspring
                offspring = []
                for j in range(lambda_):
                    if calls_used >= budget:
                        break
                    if time.time() > deadline:
                        break
                        
                    z = rng.standard_normal(d)
                    y = island['x'] + island['sigma'] * z
                    y = np.clip(y, -5.0, 5.0)
                    fy = func(y)
                    calls_used += 1
                    offspring.append((fy, y))
                    
                    if fy < best_f_global:
                        best_f_global = fy
                        best_x_global = y.copy()
                    
                    # Update island best if offspring is better
                    if offspring and fy < island['f']:
                        island['x'] = y
                        island['f'] = fy
                        # Simple sigma adaptation: increase if improved
                        island['sigma'] *= 1.05
                else:
                    # If no improvement in this island's offspring, decrease sigma
                    # Note: The 'else' here is tricky with breaks. 
                    # Let's check if the best offspring was better than the parent.
                    if offspring:
                        best_offspring_f, best_offspring_x = min(offspring, key=lambda o: o[0])
                        if best_offspring_f < island['f']:
                            island['x'] = best_offspring_x
                            island['f'] = best_offspring_f
                            island['sigma'] *= 1.05
                        else:
                            island['sigma'] *= 0.9
                    else:
                        island['sigma'] *= 0.9

            # Migration: Every few generations, share the best solution
            # Let's do it every 2 generations (approx 50 calls)
            if calls_used % 50 < n_islands * lambda_:
                # Find the best island
                best_island_idx = min(range(n_islands), key=lambda i: islands[i]['f'])
                best_island = islands[best_island_idx]
                
                # Copy best solution to other islands with some perturbation to maintain diversity
                for i in range(n_islands):
                    if i == best_island_idx:
                        continue
                    # Replace 50% of the time, or just perturb
                    if rng.random() < 0.5:
                        islands[i]['x'] = best_island['x'] + rng.standard_normal(d) * 0.5
                        islands[i]['f'] = np.inf # Force re-evaluation? No, we don't have budget for that immediately.
                        # Actually, we don't know the f of the new x without a call.
                        # To save calls, we just set the position. The next generation will evaluate offspring from here.
                        # But we need a valid 'f' for comparison in the next step?
                        # In (1+lambda), we compare offspring to parent. If parent f is inf, any finite offspring wins.
                        # This effectively resets the island to the best region.
                        # Let's keep the old f for now to avoid immediate "win" logic errors if we don't evaluate parent.
                        # Actually, standard practice: if we migrate, we should evaluate the new parent?
                        # That costs 1 call per island. 4 calls. Acceptable.
                        if calls_used < budget:
                            fx_new = func(islands[i]['x'])
                            calls_used += 1
                            islands[i]['f'] = fx_new
                            if fx_new < best_f_global:
                                best_f_global = fx_new
                                best_x_global = islands[i]['x'].copy()
                        
                        # Reset sigma to a medium value to explore from new location
                        islands[i]['sigma'] = 0.5
                    else:
                        # Keep current island, maybe increase sigma to escape local trap
                        islands[i]['sigma'] *= 1.2

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
