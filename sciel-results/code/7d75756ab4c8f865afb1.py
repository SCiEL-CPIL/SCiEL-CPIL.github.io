"""Multi-Start Nelder-Mead black-box optimizer for the rotated Sphere (BBOB f1)."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize


class Optimizer:
    """Multi-Start Nelder-Mead for 5-D continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # State variables
        best_pos = None
        best_fit = float('inf')
        calls_made = 0

        def safe_eval(x):
            nonlocal calls_made, best_pos, best_fit
            if calls_made >= budget:
                return float('inf')
            f_val = func(x)
            calls_made += 1
            if f_val < best_fit:
                best_fit = f_val
                best_pos = x.copy()
            return f_val

        # Phase 1: Coarse Global Search
        # Use 10% of budget for random sampling to find good basins
        n_global = max(10, budget // 10)
        candidates = []
        
        for _ in range(n_global):
            if calls_made >= budget:
                break
            x = rng.uniform(-5.0, 5.0, d)
            f_val = safe_eval(x)
            candidates.append((f_val, x))
            
        if calls_made >= budget:
            return best_pos, best_fit

        # Sort candidates by fitness
        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Local Refinement with Nelder-Mead
        # Pick top K candidates
        K = min(5, len(candidates))
        remaining_budget = budget - calls_made
        sub_budget_per_run = max(10, remaining_budget // K)
        
        for i in range(K):
            if calls_made >= budget:
                break
            
            # Initialize simplex around the candidate
            x0 = candidates[i][1]
            # Small perturbation for simplex vertices
            # Standard NM initialization: x0, x0 + delta * e_i
            delta = 0.5
            
            simplex = [x0.copy()]
            for j in range(d):
                x_j = x0.copy()
                x_j[j] += delta
                simplex.append(x_j)
            
            # Evaluate initial simplex
            simplex_fits = []
            for x in simplex:
                if calls_made >= budget:
                    break
                f_val = safe_eval(x)
                simplex_fits.append(f_val)
            
            if calls_made >= budget:
                break
                
            # Run Nelder-Mead iterations
            # We implement a simple NM loop
            # To avoid complex manual NM implementation errors, we use scipy.optimize.minimize
            # but we must strictly enforce the budget.
            # Since we can't easily interrupt scipy.optimize.minimize mid-evaluation without a callback that raises an exception,
            # and exceptions might be caught or cause issues, we will implement a manual simple NM.
            
            # Manual Nelder-Mead
            # Simplex is a list of points, simplex_fits is list of values
            # Sort simplex by fitness
            order = np.argsort(simplex_fits)
            simplex = [simplex[i] for i in order]
            simplex_fits = [simplex_fits[i] for i in order]
            
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            max_iter = sub_budget_per_run
            # Each iteration costs at least 1 eval (reflection).
            # Max iterations is roughly sub_budget_per_run / 2 to be safe.
            for _ in range(max_iter):
                if calls_made >= budget:
                    break
                    
                # Sort
                order = np.argsort(simplex_fits)
                simplex = [simplex[i] for i in order]
                simplex_fits = [simplex_fits[i] for i in order]
                
                x_worst = simplex[-1]
                f_worst = simplex_fits[-1]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = safe_eval(x_ref)
                
                if calls_made >= budget:
                    break
                    
                if f_ref < simplex_fits[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = safe_eval(x_exp)
                    
                    if calls_made >= budget:
                        break
                        
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_fits[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_fits[-1] = f_ref
                        
                elif f_ref < simplex_fits[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    simplex_fits[-1] = f_ref
                    
                else:
                    # Contraction
                    x_con = centroid + rho * (x_worst - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = safe_eval(x_con)
                    
                    if calls_made >= budget:
                        break
                        
                    if f_con < f_worst:
                        simplex[-1] = x_con
                        simplex_fits[-1] = f_con
                    else:
                        # Shrink
                        for j in range(1, d + 1):
                            if calls_made >= budget:
                                break
                            x_shrink = simplex[0] + sigma * (simplex[j] - simplex[0])
                            x_shrink = np.clip(x_shrink, -5.0, 5.0)
                            f_shrink = safe_eval(x_shrink)
                            simplex[j] = x_shrink
                            simplex_fits[j] = f_shrink
                            
            # Update global best is handled by safe_eval

        return best_pos, best_fit
# EVOLVE-BLOCK-END
