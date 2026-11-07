import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        # We perform multiple restarts to ensure robustness against bad initialization.
        # Given the smoothness of Sphere, 5 restarts should be plenty.
        num_restarts = 5
        budget_per_restart = budget // num_restarts
        
        for _ in range(num_restarts):
            if calls >= budget:
                break
            
            # Initialize simplex
            # Use a random center
            center = np.random.uniform(-5, 5, dim)
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = center
            
            # Perturbations
            # Ensure the simplex is not degenerate
            for i in range(1, dim + 1):
                simplex[i] = center + 1.0 * np.random.randn(dim)
                simplex[i] = np.clip(simplex[i], -5, 5)
            
            # Evaluate initial simplex
            f_values = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls >= budget:
                    break
                f_values[i] = func(simplex[i])
                calls += 1
                if f_values[i] < best_f:
                    best_f = f_values[i]
                    best_x = simplex[i].copy()
            
            if calls >= budget:
                break
            
            # Nelder-Mead Loop
            # Run until budget is nearly exhausted or no improvement
            while calls < budget:
                # Sort simplex
                idx = np.argsort(f_values)
                simplex = simplex[idx]
                f_values = f_values[idx]
                
                # Check for convergence (optional, but good for efficiency)
                # If the spread of function values is very small, stop this restart
                if np.max(f_values) - np.min(f_values) < 1e-8:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                
                if calls >= budget:
                    break
                f_r = func(x_r)
                calls += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                
                if f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if calls >= budget:
                        break
                    f_e = func(x_e)
                    calls += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    # Reflection is better than worst but not best
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_values[-1]:
                        # Outer contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inner contraction
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    if calls >= budget:
                        break
                    f_c = func(x_c)
                    calls += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    
                    if f_c < min(f_r, f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrinkage
                        for i in range(1, dim + 1):
                            if calls >= budget:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = func(simplex[i])
                            calls += 1
                            if f_values[i] < best_f:
                                best_f = f_values[i]
                                best_x = simplex[i].copy()
            
            # End of restart

        return best_x, best_f
