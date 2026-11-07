"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid (1+1)-ES with Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        # State
        best_x = None
        best_f = float('inf')
        calls_made = 0
        rng = np.random.default_rng(42)
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def check_time():
            return time.time() > deadline

        # Phase 1: (1+1)-ES for Global Exploration
        # Large initial step to cover the domain quickly given hidden shifts
        x = rng.uniform(-4, 4, dim)
        f = func(x)
        calls_made += 1
        update_best(x, f)
        
        step_size = 2.0
        # 1/5 success rule parameters
        success_count = 0
        window_size = 10
        
        es_calls_limit = int(budget * 0.3) # Use 30% of budget for exploration
        
        while calls_made < budget - 50 and not check_time(): # Leave some budget for NM
            if calls_made >= es_calls_limit:
                break
                
            # Generate candidate
            z = rng.standard_normal(dim)
            x_trial = x + step_size * z
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = func(x_trial)
            calls_made += 1
            
            if f_trial < f:
                x = x_trial
                f = f_trial
                success_count += 1
                update_best(x, f)
            else:
                # Keep old x, but step size adapts based on success rate
                pass
            
            # Adapt step size using 1/5 rule every 'window_size' steps
            # This is a simplified online version
            if (calls_made - 1) % window_size == 0 and calls_made > 1:
                # Approximate success rate over last window
                # For strict 1/5 rule we'd track history, but simple adaptation:
                # If we just improved, slightly increase, else decrease?
                # Better: Track a rolling window of successes.
                # For simplicity and speed, we use a stochastic adaptation:
                # If success in last step, multiply by 1.05, else 0.95? 
                # Let's stick to a robust simple adaptation:
                # We'll just use a fixed small increase/decrease based on immediate success 
                # to avoid complex bookkeeping, or better, use the standard 1/5 logic 
                # with a small buffer.
                
                # Let's implement a simple adaptive step size:
                # If success, step *= 1.1; else step *= 0.9
                # This is fast and effective for 1/5 rule approximation
                if f_trial < f:
                    step_size *= 1.1
                else:
                    step_size *= 0.9
                
                # Clamp step size
                step_size = np.clip(step_size, 0.01, 10.0)
            
            # Early switch to local search if we are in a promising region
            # Heuristic: If step size is small enough, we are likely near a local min
            if step_size < 0.5:
                break

        # Phase 2: Nelder-Mead for Local Refinement
        # Initialize simplex around current best
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1
            update_best(best_x, best_f)

        # Create initial simplex
        simplex = []
        f_simplex = []
        
        # Vertex 1: best_x
        simplex.append(best_x.copy())
        f_simplex.append(best_f)
        
        # Other vertices: perturb best_x
        delta = 0.5
        for i in range(dim):
            x_new = best_x.copy()
            x_new[i] += delta
            x_new = np.clip(x_new, -5, 5)
            f_new = func(x_new)
            calls_made += 1
            update_best(x_new, f_new)
            simplex.append(x_new)
            f_simplex.append(f_new)
        
        simplex = np.array(simplex)
        f_simplex = np.array(f_simplex)
        
        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrink
        
        nm_iterations = 0
        max_nm_iter = 200
        
        while calls_made < budget and not check_time():
            if nm_iterations >= max_nm_iter:
                break
                
            nm_iterations += 1
            
            # Sort simplex by fitness
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            # Check convergence
            if f_simplex[-1] - f_simplex[0] < 1e-10:
                break
            
            # Centroid of all but worst point
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # 1. Reflection
            x_reflect = x_centroid + alpha * (x_centroid - simplex[-1])
            x_reflect = np.clip(x_reflect, -5, 5)
            f_reflect = func(x_reflect)
            calls_made += 1
            update_best(x_reflect, f_reflect)
            
            if f_simplex[0] <= f_reflect < f_simplex[-2]:
                # Accept reflection
                simplex[-1] = x_reflect
                f_simplex[-1] = f_reflect
            elif f_reflect < f_simplex[0]:
                # 2. Expansion
                x_expand = x_centroid + gamma * (x_reflect - x_centroid)
                x_expand = np.clip(x_expand, -5, 5)
                f_expand = func(x_expand)
                calls_made += 1
                update_best(x_expand, f_expand)
                
                if f_expand < f_reflect:
                    simplex[-1] = x_expand
                    f_simplex[-1] = f_expand
                else:
                    simplex[-1] = x_reflect
                    f_simplex[-1] = f_reflect
            else:
                # 3. Contraction
                if f_reflect < f_simplex[-1]:
                    # Outside contraction
                    x_contract = x_centroid + rho * (x_reflect - x_centroid)
                else:
                    # Inside contraction
                    x_contract = x_centroid - rho * (x_reflect - x_centroid)
                
                x_contract = np.clip(x_contract, -5, 5)
                f_contract = func(x_contract)
                calls_made += 1
                update_best(x_contract, f_contract)
                
                if f_contract < min(f_reflect, f_simplex[-1]):
                    simplex[-1] = x_contract
                    f_simplex[-1] = f_contract
                else:
                    # 4. Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_simplex[i] = func(simplex[i])
                        calls_made += 1
                        update_best(simplex[i], f_simplex[i])
                    
                    if check_time() or calls_made >= budget:
                        break

        # Final safety check
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
