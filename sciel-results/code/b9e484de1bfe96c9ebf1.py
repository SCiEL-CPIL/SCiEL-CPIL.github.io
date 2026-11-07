# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Nelder-Mead optimizer for black-box continuous functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        calls_made = 0
        best_x = None
        best_f = float('inf')

        def eval_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return float('inf')
            calls_made += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Initialize simplex
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-5, 5, n)
        
        # Simplex vertices
        # Start with a larger initial step to ensure we cover the potential valley
        delta = 1.0 
        simplex = [x0]
        for i in range(n):
            v = x0.copy()
            v[i] += delta
            simplex.append(v)
        
        # Evaluate initial simplex
        f_simplex = [eval_func(v) for v in simplex]
        
        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        while calls_made < self.budget:
            # Sort simplex by function value
            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]
            
            # Check convergence
            if np.max(f_simplex) - np.min(f_simplex) < 1e-10:
                break
                
            # Centroid of best n points
            centroid = np.mean(simplex[:n], axis=0)
            
            # Worst point
            x_worst = simplex[-1]
            
            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            f_r = eval_func(x_r)
            
            if f_r < f_simplex[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - x_worst)
                f_e = eval_func(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_simplex[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
            elif f_r < f_simplex[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_simplex[-1] = f_r
            else:
                # Contraction
                # Outside contraction
                x_c = centroid + rho * (x_r - centroid)
                f_c = eval_func(x_c)
                
                if f_c < f_r:
                    simplex[-1] = x_c
                    f_simplex[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        f_simplex[i] = eval_func(simplex[i])
            
        return best_x, best_f
# EVOLVE-BLOCK-END
