"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Deterministic Nelder-Mead Simplex with coarse grid seeding for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State
        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)

        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget or time.time() > deadline:
                return None
            x = np.clip(x, -5, 5)
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse Grid Exploration
        # For 5D, a grid of 3 points per dimension is 3^5 = 243 points.
        # This is too many if we want to save budget for refinement, but 
        # BBOB f22 is multimodal. Let's use a slightly sparser grid or random.
        # Actually, 500 calls is tight. 
        # Strategy: 
        # 1. Evaluate a set of diverse starting points.
        # 2. Run Nelder-Mead from the best few points.
        
        # Let's use a grid of 2 points per dimension? 2^5 = 32 points.
        # Or 3 points per dimension for key axes?
        # Let's try a stratified random sampling + origin.
        
        # Initial Points:
        # 1. Origin
        # 2. 10 random points in [-4, 4]
        # 3. 5 points along axes (positive and negative)
        
        initial_points = []
        initial_points.append(np.zeros(dim))
        
        # Random points
        for _ in range(10):
            initial_points.append(rng.uniform(-4, 4, size=dim))
            
        # Axis points
        for i in range(dim):
            for val in [-3, 3]:
                p = np.zeros(dim)
                p[i] = val
                initial_points.append(p)
                
        # Evaluate all initial points
        for p in initial_points:
            f = do_call(p)
            if f is None:
                break

        if call_count >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Nelder-Mead Local Optimization
        # We will run Nelder-Mead starting from the current best point.
        # Since we don't have a list of all evaluated points sorted, 
        # we just use the global best found so far.
        # To be more robust, we could run NM from the top 2-3 best points,
        # but we didn't store them. Let's just use the global best.
        # Given the budget, one good NM run is better than many bad ones.
        
        # Nelder-Mead Parameters
        # Initial simplex: best_x + alpha * basis
        alpha = 0.1
        # Simplex size relative to domain
        # Let's start with a moderately sized simplex
        initial_step = 0.5
        
        # Construct initial simplex
        simplex = [best_x.copy()]
        for i in range(dim):
            v = best_x.copy()
            v[i] += initial_step
            simplex.append(v)
            
        # Evaluate simplex vertices
        simplex_f = []
        for v in simplex:
            f = do_call(v)
            if f is None:
                return best_x, best_f
            simplex_f.append(f)
            
        # Sort simplex by function value
        # We need to keep track of points and values together
        # Create list of (f, x)
        pop = list(zip(simplex_f, simplex))
        pop.sort(key=lambda item: item[0])
        
        # Nelder-Mead Loop
        # Standard parameters
        alpha_n = 1.0  # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma = 0.5    # shrink
        
        # Max iterations for NM to prevent infinite loops or wasting budget
        # Each iteration costs ~dim+1 calls on average, but usually 1-2.
        # Let's allow up to 200 NM iterations or until budget is low.
        max_nm_iter = 200
        
        for _ in range(max_nm_iter):
            if call_count + dim + 2 > budget or time.time() > deadline:
                break
                
            # Sort population
            pop.sort(key=lambda item: item[0])
            
            # If the best point is very close to 0 or we have converged, break?
            # Check convergence: difference between best and worst
            f_best = pop[0][0]
            f_worst = pop[-1][0]
            if abs(f_worst - f_best) < 1e-6:
                break
            
            # Worst point
            x_worst = pop[-1][1]
            f_worst_val = pop[-1][0]
            
            # Centroid of all points except the worst
            x_centroid = np.mean([p[1] for p in pop[:-1]], axis=0)
            
            # 1. Reflection
            x_ref = x_centroid + alpha_n * (x_centroid - x_worst)
            f_ref = do_call(x_ref)
            if f_ref is None:
                break
            
            if f_ref < pop[0][0]:
                # Reflection is better than best
                # 2. Expansion
                x_exp = x_centroid + gamma * (x_centroid - x_worst)
                f_exp = do_call(x_exp)
                if f_exp is None:
                    break
                
                if f_exp < f_ref:
                    # Replace worst with expansion
                    pop[-1] = (f_exp, x_exp)
                else:
                    # Replace worst with reflection
                    pop[-1] = (f_ref, x_ref)
            elif f_ref < f_worst_val:
                # Reflection is better than worst but not best
                # Replace worst with reflection
                pop[-1] = (f_ref, x_ref)
            else:
                # Reflection is worse than worst
                # 3. Contraction
                x_con = x_centroid + rho * (x_worst - x_centroid)
                f_con = do_call(x_con)
                if f_con is None:
                    break
                
                if f_con < f_worst_val:
                    # Replace worst with contraction
                    pop[-1] = (f_con, x_con)
                else:
                    # 4. Shrink
                    # Move all points (except best) halfway to best
                    x_best = pop[0][1]
                    for i in range(1, len(pop)):
                        x_shrink = x_best + sigma * (pop[i][1] - x_best)
                        f_shrink = do_call(x_shrink)
                        if f_shrink is None:
                            return best_x, best_f
                        pop[i] = (f_shrink, x_shrink)
                    
                    # Re-sort after shrink
                    pop.sort(key=lambda item: item[0])
                    
                    # Update global best if needed (do_call already does this)

        # Phase 3: Final Polish
        # If we have budget left, do a few random perturbations around the best point
        if call_count < budget and time.time() < deadline:
            remaining = budget - call_count
            n_polish = min(remaining, 20)
            for _ in range(n_polish):
                if call_count >= budget or time.time() > deadline:
                    break
                x_trial = best_x + rng.normal(0, 0.01, size=dim)
                do_call(x_trial)

        return best_x, best_f

# EVOLVE-BLOCK-END
