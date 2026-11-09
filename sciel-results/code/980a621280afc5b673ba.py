# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead Simplex Method with restarts for ridge functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            if calls_made[0] >= budget:
                raise Exception("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            calls_made[0] += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(12345)
        
        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        # Initial simplex size
        init_sigma = 1.0
        
        def run_nelder_mead(start_x, max_iter):
            """Run Nelder-Mead from start_x for up to max_iter evaluations."""
            nonlocal calls_made, best_f, best_x, deadline
            
            # Initialize simplex
            # n+1 points
            simplex = [start_x.copy()]
            for i in range(n):
                pt = start_x.copy()
                # Use a small step in each dimension
                # If start_x is near boundary, adjust step
                step = init_sigma
                if start_x[i] + step > 5:
                    pt[i] = start_x[i] - step
                elif start_x[i] - step < -5:
                    pt[i] = start_x[i] + step
                else:
                    pt[i] = start_x[i] + step
                simplex.append(pt)
            
            # Evaluate initial simplex
            f_values = [wrapped_func(pt) for pt in simplex]
            
            if calls_made[0] >= budget:
                return
            
            iter_count = 0
            stall_count = 0
            prev_best = best_f
            
            while calls_made[0] < budget and iter_count < max_iter:
                if time.time() > deadline:
                    break
                
                # Sort simplex by fitness
                sorted_indices = np.argsort(f_values)
                simplex = [simplex[i] for i in sorted_indices]
                f_values = [f_values[i] for i in sorted_indices]
                
                # Check for convergence/stall
                if f_values[0] < prev_best - 1e-10:
                    prev_best = f_values[0]
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count > 20:
                        break
                
                # Get worst and best
                x_worst = simplex[-1]
                f_worst = f_values[-1]
                x_best = simplex[0]
                
                # Compute centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = wrapped_func(x_r)
                
                if calls_made[0] >= budget:
                    break
                
                if f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - x_worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = wrapped_func(x_e)
                    
                    if calls_made[0] >= budget:
                        break
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    # Simple reflection
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = wrapped_func(x_c)
                    
                    if calls_made[0] >= budget:
                        break
                    
                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = wrapped_func(simplex[i])
                        
                        if calls_made[0] >= budget:
                            break
                        
                        # Check if simplex is too small
                        simplex_size = np.max([np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, n+1)])
                        if simplex_size < 1e-6:
                            break
                
                iter_count += 1

        # Multi-start strategy
        # Start from center, then a few random points if budget allows
        start_points = [np.zeros(n)]
        # Add a few random starts to escape local minima
        for _ in range(4):
            start_points.append(rng.uniform(-4, 4, n))
        
        # Allocate budget: 60% to first start, 10% to others
        # Or just run sequentially until budget is exhausted
        # Since Nelder-Mead is efficient, we can run one instance for a while,
        # then restart from the best found so far with a perturbation.
        
        current_best_x = best_x.copy()
        current_best_f = best_f
        
        # Run first instance
        run_nelder_mead(np.zeros(n), max_iter=100)
        
        # If we still have budget, restart from best with perturbation
        while calls_made[0] < budget:
            if time.time() > deadline:
                break
            
            # Perturb best point
            perturbation = rng.standard_normal(n) * 0.5
            new_start = current_best_x + perturbation
            new_start = np.clip(new_start, -5, 5)
            
            # Run Nelder-Mead from this new start
            # Limit iterations to avoid getting stuck in a local basin
            run_nelder_mead(new_start, max_iter=50)
            
            # Update best
            if best_f < current_best_f:
                current_best_f = best_f
                current_best_x = best_x.copy()
            else:
                # If no improvement, increase perturbation size slightly
                # Or just break if no improvement in several restarts?
                # Let's just continue, the perturbation might work next time.
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
