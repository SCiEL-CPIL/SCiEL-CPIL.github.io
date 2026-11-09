import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead optimizer for anisotropic functions like Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin of 10s
        
        rng = np.random.default_rng(42)
        
        # We will use a multi-start approach.
        # Bent Cigar is unimodal, but the hidden shift/rotation/offset means the "cigar" 
        # might be oriented arbitrarily. Nelder-Mead adapts to this.
        # Let's use 5 restarts, sharing the budget.
        n_restarts = 5
        budget_per_restart = self.budget // n_restarts
        
        best_x = None
        best_f = np.inf

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if self.budget - (r * budget_per_restart) <= 0:
                break

            # Initialize simplex
            # Start point
            x0 = rng.uniform(-4, 4, self.dim)
            
            # Create initial simplex
            # The simplex size should be moderate. 
            # For [-5, 5]^5, a size of 0.5 is a good start.
            simplex = [x0]
            for i in range(self.dim):
                x_i = x0.copy()
                x_i[i] += 0.5
                # Ensure within bounds
                x_i = np.clip(x_i, -5, 5)
                simplex.append(x_i)
            
            # Evaluate initial simplex
            f_simplex = [func(x) for x in simplex]
            
            # Track best so far
            current_best_idx = np.argmin(f_simplex)
            if f_simplex[current_best_idx] < best_f:
                best_f = f_simplex[current_best_idx]
                best_x = simplex[current_best_idx].copy()

            # Nelder-Mead parameters
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrinkage
            
            iterations_in_restart = 0
            max_iter_restart = budget_per_restart - (self.dim + 1)
            
            # Main Nelder-Mead loop
            while iterations_in_restart < max_iter_restart:
                if time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Check convergence
                if np.max(f_simplex) - np.min(f_simplex) < 1e-8:
                    break
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                
                if fr < f_simplex[0]:
                    # Try expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                elif fr < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                else:
                    # Contraction
                    # Outside contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                
                iterations_in_restart += 1
                
                # Update global best
                current_min_in_simplex = np.min(f_simplex)
                if current_min_in_simplex < best_f:
                    best_f = current_min_in_simplex
                    best_idx = np.argmin(f_simplex)
                    best_x = simplex[best_idx].copy()

        return best_x, best_f
