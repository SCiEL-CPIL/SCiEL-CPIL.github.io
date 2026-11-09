"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead optimizer with boundary handling."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrinkage

        # Initialize simplex
        # Start at center (0,0,0,0,0)
        # Create dim+1 points
        # Point 0 is the center
        # Points 1..dim are offset along axes
        init_radius = 2.0
        
        simplex = np.zeros((dim + 1, dim))
        for i in range(dim):
            simplex[i + 1, i] = init_radius
            
        # Evaluate initial simplex
        func_values = np.zeros(dim + 1)
        calls_made = 0
        
        for i in range(dim + 1):
            x = np.clip(simplex[i], -5, 5)
            func_values[i] = func(x)
            calls_made += 1
            if calls_made >= budget:
                break

        # Sort simplex by function value
        indices = np.argsort(func_values)
        simplex = simplex[indices]
        func_values = func_values[indices]

        best_x = simplex[0].copy()
        best_f = func_values[0]

        # Main loop
        while calls_made < budget:
            # Compute centroid of all but the worst point
            centroid = np.sum(simplex[:dim], axis=0) / dim
            
            # 1. Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_made += 1
            
            if calls_made >= budget:
                break

            if f_r < func_values[0]:
                # Try Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls_made += 1
                
                if calls_made >= budget:
                    break

                if f_e < f_r:
                    simplex[-1] = x_e
                    func_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    func_values[-1] = f_r
            else:
                if f_r < func_values[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    func_values[-1] = f_r
                else:
                    # Try Contraction
                    # Outer contraction if f_r < func_values[-1] (which is true by previous else)
                    # Wait, standard NM:
                    # If f_r >= f_w (worst):
                    #   If f_r < f_{n-1} (second worst): Outer Contraction
                    #   Else: Inner Contraction
                    if f_r < func_values[-1]:
                        # Outer Contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inner Contraction
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    
                    if calls_made >= budget:
                        break

                    if f_c < min(f_r, func_values[-1]):
                        simplex[-1] = x_c
                        func_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            func_values[i] = func(simplex[i])
                            calls_made += 1
                            if calls_made >= budget:
                                break
                        if calls_made >= budget:
                            break

            # Re-sort simplex
            indices = np.argsort(func_values)
            simplex = simplex[indices]
            func_values = func_values[indices]
            
            # Update best
            if func_values[0] < best_f:
                best_f = func_values[0]
                best_x = simplex[0].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
