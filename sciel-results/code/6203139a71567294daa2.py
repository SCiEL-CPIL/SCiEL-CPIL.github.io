"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead Simplex Optimizer with Multi-Start and Adaptive Restart."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink

        def nelder_mead(start_x, remaining_budget, initial_step=3.0):
            nonlocal calls_made, best_f, best_x
            
            if remaining_budget <= 0:
                return best_x, best_f

            # Initialize Simplex
            # Vertices: start_x and start_x + step * e_i
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = start_x
            for i in range(dim):
                simplex[i + 1] = start_x.copy()
                simplex[i + 1][i] += initial_step
            
            # Evaluate initial simplex
            simplex_f = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls_made >= budget:
                    break
                x_i = np.clip(simplex[i], -5, 5)
                simplex_f[i] = func(x_i)
                calls_made += 1
                simplex[i] = x_i # Keep the clipped version for geometry consistency? 
                                 # Actually, better to keep the original for step calculations 
                                 # but evaluate the clipped one. 
                                 # If we clip, the geometry is distorted. 
                                 # Let's just evaluate the clipped point and store the clipped point 
                                 # to maintain a valid simplex within the box.
                
                if simplex_f[i] < best_f:
                    best_f = simplex_f[i]
                    best_x = x_i.copy()

            if calls_made >= budget:
                return best_x, best_f

            # Main Loop
            while calls_made < budget:
                # Sort vertices by function value
                idx = np.argsort(simplex_f)
                simplex = simplex[idx]
                simplex_f = simplex_f[idx]
                
                # Check for convergence (simplex size)
                # Calculate distance between best and worst
                dist = np.linalg.norm(simplex[0] - simplex[-1])
                if dist < 1e-6 or np.allclose(simplex_f, simplex_f[0], atol=1e-8):
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # 1. Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_made += 1
                
                if f_r < simplex_f[0]:
                    # 2. Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_made += 1
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Reflection is good
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # 3. Contraction
                    # Outside contraction if f_r >= f_w, Inside if f_r < f_w
                    if f_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    
                    if f_c < min(f_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # 4. Shrink
                        # Shrink all vertices except the best towards the best
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls_made >= budget:
                                break
                            simplex_f[i] = func(simplex[i])
                            calls_made += 1
                        if calls_made >= budget:
                            break

                # Update global best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()

            return best_x, best_f

        # Strategy: 
        # 1. Start with a large simplex at the origin (0,0,0,0,0)
        # 2. If budget remains, restart from the best point with a smaller or similar step?
        #    Or just run one long Nelder-Mead. 
        #    Nelder-Mead is local. If the function is convex (Linear Slope is convex), 
        #    one run should suffice if the initial simplex is large enough to cover the basin.
        #    Step size 3.0 in 5D with box [-5,5] is reasonable.
        
        nelder_mead(np.zeros(dim), budget, initial_step=4.0)
        
        # If we have budget left, we can try a second run from a random point 
        # or refine the best point.
        # Given the "Linear Slope" nature, the function is likely convex. 
        # One run should be enough. But to be safe, if we have > 50 calls left,
        # we can do a fine-tune.
        
        if calls_made < budget and best_x is not None:
             # Run a second pass with a smaller step size centered at best_x
             nelder_mead(best_x, budget - calls_made, initial_step=1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
