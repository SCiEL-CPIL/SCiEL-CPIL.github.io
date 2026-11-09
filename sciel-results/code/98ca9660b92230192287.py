"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead based continuous optimizer with multi-start."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        # Multi-start configuration
        num_starts = 8
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(n)
        best_f = float('inf')
        
        # Generate start points in [-4, 4]^5
        start_points = rng.uniform(-4.0, 4.0, size=(num_starts, n))
        
        evals_used = 0
        
        for s_idx in range(num_starts):
            if evals_used >= budget:
                break
            
            x_start = start_points[s_idx]
            
            # Initialize simplex
            delta = 1.0
            simplex = np.zeros((n + 1, n))
            simplex[0] = x_start
            for i in range(n):
                simplex[i + 1] = x_start
                simplex[i + 1, i] += delta
            
            # Evaluate initial simplex
            f_values = np.zeros(n + 1)
            for i in range(n + 1):
                if evals_used >= budget:
                    break
                x = np.clip(simplex[i], -5.0, 5.0)
                simplex[i] = x
                f_values[i] = func(x)
                evals_used += 1
                
                if f_values[i] < best_f:
                    best_f = f_values[i]
                    best_x = x.copy()
            
            if evals_used >= budget:
                break
                
            # Nelder-Mead iterations
            # Limit iterations to prevent infinite loops on flat regions
            max_iter = 50 
            for _ in range(max_iter):
                if evals_used >= budget:
                    break
                    
                # Sort simplex by function value
                order = np.argsort(f_values)
                simplex = simplex[order]
                f_values = f_values[order]
                
                # Check convergence
                if np.max(f_values) - np.min(f_values) < 1e-8:
                    break
                    
                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflect
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5.0, 5.0)
                
                if evals_used >= budget:
                    break
                f_reflect = func(x_reflect)
                evals_used += 1
                
                if f_reflect < f_values[0]:
                    # Try expansion
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5.0, 5.0)
                    
                    if evals_used >= budget:
                        break
                    f_expand = func(x_expand)
                    evals_used += 1
                    
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        f_values[-1] = f_expand
                        if f_expand < best_f:
                            best_f = f_expand
                            best_x = x_expand.copy()
                    else:
                        simplex[-1] = x_reflect
                        f_values[-1] = f_reflect
                        if f_reflect < best_f:
                            best_f = f_reflect
                            best_x = x_reflect.copy()
                        
                elif f_reflect < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    f_values[-1] = f_reflect
                    if f_reflect < best_f:
                        best_f = f_reflect
                        best_x = x_reflect.copy()
                        
                else:
                    # Contraction
                    if f_reflect < f_values[-1]:
                        x_contract = centroid + rho * (x_reflect - centroid)
                    else:
                        x_contract = centroid - rho * (x_reflect - centroid)
                    x_contract = np.clip(x_contract, -5.0, 5.0)
                    
                    if evals_used >= budget:
                        break
                    f_contract = func(x_contract)
                    evals_used += 1
                    
                    if f_contract < f_reflect:
                        simplex[-1] = x_contract
                        f_values[-1] = f_contract
                        if f_contract < best_f:
                            best_f = f_contract
                            best_x = x_contract.copy()
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            
                            if evals_used >= budget:
                                break
                            f_values[i] = func(simplex[i])
                            evals_used += 1
                            
                            if f_values[i] < best_f:
                                best_f = f_values[i]
                                best_x = simplex[i].copy()
                        
                        if evals_used >= budget:
                            break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
