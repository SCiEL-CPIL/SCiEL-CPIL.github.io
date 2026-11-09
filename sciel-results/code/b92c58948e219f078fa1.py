"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid Coordinate Descent + Nelder-Mead Optimizer.
    
    Strategy:
    1. Use multiple restarts with diverse initializations.
    2. For each restart, alternate between coordinate descent (fast local exploitation)
       and Nelder-Mead simplex (handling non-separable functions like Rosenbrock/Rastrigin).
    3. Adaptive step sizes that shrink as progress is made.
    4. Strict budget and time management.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s, keep 10s buffer
        deadline = start_time + 320.0 
        
        best_x = None
        best_f = float('inf')
        
        # We have 300 evaluations total.
        # Strategy: Split budget among restarts.
        # Early restarts get larger budgets to find the basin.
        # Later restarts get smaller budgets to polish.
        
        # Define restart schedule:
        # Restart 0: 100 evals (broad search)
        # Restart 1: 80 evals
        # Restart 2: 60 evals
        # Restart 3: 40 evals
        # Restart 4: 20 evals
        # Total: 300
        # If we find a very good solution early, we can stop or shift budget.
        
        rng = np.random.default_rng(123)
        
        # List of initial points to try
        # 1. Zero (often good for Sphere/Rosenbrock if shift is small)
        # 2. Random in [-4, 4]
        # 3. Random in [-2, 2]
        # 4. Best found so far + noise
        
        max_restarts = 5
        budgets = [100, 80, 60, 40, 20]
        
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            current_budget = budgets[r]
            if current_budget <= 0:
                break
                
            # Determine starting point
            if r == 0:
                x_start = np.zeros(self.dim)
            elif r == 1:
                x_start = rng.uniform(-4, 4, self.dim)
            elif r == 2:
                x_start = rng.uniform(-2, 2, self.dim)
            else:
                if best_x is not None:
                    # Perturb best
                    x_start = np.clip(best_x + rng.normal(0, 0.5, self.dim), -5, 5)
                else:
                    x_start = rng.uniform(-5, 5, self.dim)
            
            # Evaluate start
            f_start = func(x_start)
            if f_start < best_f:
                best_f = f_start
                best_x = x_start.copy()
            current_budget -= 1
            
            # Run local search from x_start
            # We will use a simple loop:
            # Phase 1: Coordinate Descent with adaptive step
            # Phase 2: Nelder-Mead for final polish
            
            x = x_start.copy()
            f_x = f_start
            
            # Phase 1: Coordinate Descent
            # Use a step size that starts moderately large and shrinks
            step = 2.0
            min_step = 1e-6
            max_cd_iters = current_budget // 2  # Reserve half for NM
            
            for _ in range(max_cd_iters):
                if time.time() > deadline:
                    break
                if current_budget <= 0:
                    break
                    
                improved = False
                for i in range(self.dim):
                    if time.time() > deadline:
                        break
                    if current_budget <= 0:
                        break
                    
                    # Try +step
                    x_plus = x.copy()
                    x_plus[i] = np.clip(x_plus[i] + step, -5, 5)
                    f_plus = func(x_plus)
                    current_budget -= 1
                    
                    if f_plus < f_x:
                        x = x_plus
                        f_x = f_plus
                        improved = True
                    else:
                        # Try -step
                        x_minus = x.copy()
                        x_minus[i] = np.clip(x_minus[i] - step, -5, 5)
                        f_minus = func(x_minus)
                        current_budget -= 1
                        
                        if f_minus < f_x:
                            x = x_minus
                            f_x = f_minus
                            improved = True
                            # If we moved in negative direction, keep step size? 
                            # Or shrink? Standard CD often shrinks if no improvement, 
                            # but here we improved, so keep step.
                        else:
                            # No improvement in this coordinate, shrink step for this coordinate?
                            # Global step shrinkage is simpler.
                            pass
                
                if not improved:
                    # Shrink step size
                    step *= 0.5
                    if step < min_step:
                        break
                else:
                    # If we improved, we might keep step or shrink slightly to avoid overshooting?
                    # For smooth functions, shrinking is safe. For Rastrigin, we might need to jump.
                    # Let's shrink slowly if improved to ensure convergence.
                    step *= 0.8
                    if step < min_step:
                        break
                
                # Update global best
                if f_x < best_f:
                    best_f = f_x
                    best_x = x.copy()
                
                # Early stopping if very precise
                if abs(f_x - best_f) < 1e-10 and step < 1e-4:
                    break

            # Phase 2: Nelder-Mead Simplex
            # Use remaining budget
            if current_budget > 0:
                x = best_x.copy()
                f_x = best_f
                
                # Initialize simplex
                # Start with best point and perturbations
                simplex = [x.copy()]
                f_values = [f_x]
                
                # Create initial vertices
                for i in range(self.dim):
                    v = x.copy()
                    # Perturb in direction i
                    v[i] += 0.1 * (1.0 if v[i] < 4.9 else -1.0) # Stay in bounds
                    v[i] = np.clip(v[i], -5, 5)
                    f_v = func(v)
                    current_budget -= 1
                    simplex.append(v)
                    f_values.append(f_v)
                    if f_v < best_f:
                        best_f = f_v
                        best_x = v.copy()
                
                # Nelder-Mead Loop
                # Standard NM uses reflection, expansion, contraction, shrink
                # We need to be careful with budget.
                # Each iteration uses 1 evaluation (reflection).
                # Max iterations = current_budget
                
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5
                
                for _ in range(current_budget):
                    if time.time() > deadline:
                        break
                    
                    # Sort simplex by function value
                    order = np.argsort(f_values)
                    simplex = [simplex[i] for i in order]
                    f_values = [f_values[i] for i in order]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5, 5)
                    f_r = func(x_r)
                    current_budget -= 1
                    
                    if f_r < f_values[0]:
                        # Expansion
                        x_e = centroid + gamma * (centroid - simplex[-1])
                        x_e = np.clip(x_e, -5, 5)
                        f_e = func(x_e)
                        current_budget -= 1
                        
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_values[-1] = f_e
                            if f_e < best_f:
                                best_f = f_e
                                best_x = x_e.copy()
                        else:
                            simplex[-1] = x_r
                            f_values[-1] = f_r
                            if f_r < best_f:
                                best_f = f_r
                                best_x = x_r.copy()
                    elif f_r < f_values[-2]:
                        # Reflection is good
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                        if f_r < best_f:
                            best_f = f_r
                            best_x = x_r.copy()
                    else:
                        # Contraction
                        x_c = centroid + rho * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        current_budget -= 1
                        
                        if f_c < f_values[-1]:
                            simplex[-1] = x_c
                            f_values[-1] = f_c
                            if f_c < best_f:
                                best_f = f_c
                                best_x = x_c.copy()
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_s = func(simplex[i])
                                current_budget -= 1
                                f_values[i] = f_s
                                if f_s < best_f:
                                    best_f = f_s
                                    best_x = simplex[i].copy()
                
                # Check if we ran out of budget
                if current_budget <= 0:
                    break

            # If best_f is very small, we might stop early to save time for other instances?
            # No, we only have one instance per __call__.
            # But if we are very close to optimum, further restarts might not help.
            if best_f < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
