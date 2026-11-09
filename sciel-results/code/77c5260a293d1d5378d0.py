"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Nelder-Mead with Random Restarts for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        def nelder_mead(start_x, max_calls):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return
            
            # Initialize simplex
            simplex = []
            f_vals = []
            
            # First point is start_x
            x0 = np.clip(start_x, -5, 5)
            f0 = safe_func(x0)
            simplex.append(x0)
            f_vals.append(f0)
            
            # Other points
            for i in range(dim):
                xi = x0.copy()
                # Perturb one dimension
                step = 0.5 if xi[i] < 0 else -0.5 # Move towards center or out? 
                # Better: small step in positive direction, then clip
                xi[i] += 0.5
                xi = np.clip(xi, -5, 5)
                fi = safe_func(xi)
                simplex.append(xi)
                f_vals.append(fi)
                if calls >= budget:
                    return

            # Main loop
            while calls < budget and (budget - calls) > 0:
                if len(f_vals) < dim + 1:
                    break
                
                # Sort simplex by function value
                indices = np.argsort(f_vals)
                simplex = [simplex[i] for i in indices]
                f_vals = [f_vals[i] for i in indices]
                
                # If no improvement in this iteration, break? 
                # Or just continue until max_calls or budget.
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_func(x_r)
                if calls >= budget:
                    break
                
                if f_r < f_vals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_func(x_e)
                    if calls >= budget:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_vals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = safe_func(x_c)
                    if calls >= budget:
                        break
                    if f_c < min(f_r, f_vals[-1]):
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_vals[i] = safe_func(simplex[i])
                            if calls >= budget:
                                return

        # Main strategy: Hybrid Local-Global
        # 1. Start with a random point
        current_x = rng.uniform(-5, 5, dim)
        
        # Local budget per restart
        # With 500 calls, and dim=5, NM uses ~5-10 calls per iteration.
        # Let's allow ~20-30 calls per local run.
        local_budget = 25
        
        while calls < budget:
            # Run local search from current_x
            start_calls = calls
            nelder_mead(current_x, local_budget)
            
            # Check improvement
            if calls - start_calls == 0:
                break
                
            # If we made good progress, continue from best.
            # If we made little progress, restart randomly.
            # Heuristic: if best_f hasn't changed significantly in the last local run, restart.
            # We don't track 'previous best' easily here, so let's just alternate or use a simple rule.
            # Rule: Every 3rd run, force a random restart to escape local minima.
            # Or: if the local run didn't improve best_f, restart.
            
            # Simpler: Always restart from a random point if the local run was short or unproductive?
            # Let's use a counter.
            
            # To be safe and explore, let's do a random restart every few iterations.
            # But f24 has a global min that might be far.
            # Let's try: 50% chance to restart from random, 50% from current best.
            # Actually, for BBOB, random restarts are very effective.
            
            if rng.random() < 0.5:
                current_x = rng.uniform(-5, 5, dim)
            else:
                # Perturb current best slightly
                current_x = best_x + rng.normal(0, 0.5, dim)
                current_x = np.clip(current_x, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
