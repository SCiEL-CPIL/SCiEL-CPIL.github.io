"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def check_deadline():
            return time.time() >= deadline
        
        def budget_exhausted():
            return calls_used >= self.budget
        
        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Phase 1: Random restarts with Nelder-Mead
        max_restarts = 10
        simplex_size = 1.0
        
        for restart in range(max_restarts):
            if budget_exhausted() or check_deadline():
                break
            
            # Initialize simplex around a random point
            center = rng.uniform(-4, 4, self.dim)
            simplex = []
            f_values = []
            
            # Initial vertex
            simplex.append(center.copy())
            f_values.append(evaluate(center))
            
            # Other vertices
            for i in range(self.dim):
                v = center.copy()
                v[i] += simplex_size
                simplex.append(v)
                f_values.append(evaluate(v))
            
            if budget_exhausted() or check_deadline():
                break
            
            # Nelder-Mead iterations
            alpha = 1.0    # reflection
            gamma = 2.0    # expansion
            rho = 0.5      # contraction
            sigma = 0.5    # shrink
            
            simplex = np.array(simplex)
            f_values = np.array(f_values)
            
            nm_iter = 0
            max_nm_iter = 500
            
            while nm_iter < max_nm_iter:
                if budget_exhausted() or check_deadline():
                    break
                nm_iter += 1
                
                # Sort simplex
                idx = np.argsort(f_values)
                simplex = simplex[idx]
                f_values = f_values[idx]
                
                # Check convergence
                if np.max(f_values) - np.min(f_values) < 1e-10:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                f_ref = evaluate(x_ref)
                
                if f_ref < f_values[0]:
                    # Expansion
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    f_exp = evaluate(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                elif f_ref < f_values[-2]:
                    # Reflection accepted
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_values[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[-1] - centroid)
                    f_con = evaluate(x_con)
                    
                    if f_con < min(f_ref, f_values[-1]):
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_values[i] = evaluate(simplex[i])
                
                if budget_exhausted() or check_deadline():
                    break
            
            # If we made good progress, reduce simplex size for next restart
            if restart > 0:
                simplex_size *= 0.5
        
        # Phase 2: Local refinement around best point
        if best_x is not None and not budget_exhausted() and not check_deadline():
            # Use a smaller simplex around the best point
            local_simplex_size = 0.1
            center = best_x.copy()
            simplex = []
            f_values = []
            
            simplex.append(center.copy())
            f_values.append(evaluate(center))
            
            for i in range(self.dim):
                v = center.copy()
                v[i] += local_simplex_size
                simplex.append(v)
                f_values.append(evaluate(v))
            
            if budget_exhausted() or check_deadline():
                return best_x, best_f
            
            simplex = np.array(simplex)
            f_values = np.array(f_values)
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            nm_iter = 0
            max_nm_iter = 200
            
            while nm_iter < max_nm_iter:
                if budget_exhausted() or check_deadline():
                    break
                nm_iter += 1
                
                idx = np.argsort(f_values)
                simplex = simplex[idx]
                f_values = f_values[idx]
                
                if np.max(f_values) - np.min(f_values) < 1e-12:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                x_ref = centroid + alpha * (centroid - simplex[-1])
                f_ref = evaluate(x_ref)
                
                if f_ref < f_values[0]:
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    f_exp = evaluate(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                elif f_ref < f_values[-2]:
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                else:
                    if f_ref < f_values[-1]:
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        x_con = centroid - rho * (simplex[-1] - centroid)
                    f_con = evaluate(x_con)
                    
                    if f_con < min(f_ref, f_values[-1]):
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_values[i] = evaluate(simplex[i])
        
        return best_x, best_f
# EVOLVE-BLOCK-END
