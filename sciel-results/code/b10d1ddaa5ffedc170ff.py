# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Nelder-Mead Simplex Optimizer with Latin Hypercube Initialization.
    
    Strategy:
    1. Initialization: Sample 10 points using Latin Hypercube Sampling (LHS) to ensure
       good coverage of the 5D space [-5, 5]^5. This is more efficient than pure random
       sampling for finding a promising basin in low dimensions.
    2. Optimization: Select the best initial point and initialize a Nelder-Mead simplex
       around it. Run the Nelder-Mead algorithm to converge to the local minimum.
       Nelder-Mead is derivative-free and robust for smooth functions, often outperforming
       (1+1)-ES on functions with well-defined basins like BBOB f13 (Sharp Ridge) because
       it adapts the shape of the search region rather than just the step size.
    3. Budget Management: Monitor function calls and wall-clock time strictly.
       If budget or time runs out, return the best point found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def timed_call(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= self.budget:
                return None
            if time.time() > deadline - 2:
                return None
            # Ensure x is a numpy array of floats
            x = np.asarray(x, dtype=float)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initialization with Latin Hypercube Sampling
        # We sample 10 points. LHS ensures that in each dimension, the points
        # are stratified into 10 bins, providing better coverage than uniform random.
        n_init = 10
        dim = self.dim
        
        # Generate LHS points
        # For each dimension, shuffle the indices 0..n_init-1 and map to [-5, 5]
        lhs_points = []
        for i in range(n_init):
            point = []
            for d in range(dim):
                # This simple loop isn't true LHS because it shuffles independently for each point?
                # No, standard LHS generates a permutation matrix.
                # Let's do it properly:
                pass
        
        # Proper LHS generation
        # Create a matrix of shape (dim, n_init) where each row is a permutation of 0..n_init-1
        perm_matrix = np.zeros((dim, n_init), dtype=int)
        for d in range(dim):
            perm_matrix[d, :] = rng.permutation(n_init)
        
        # Map to [-5, 5]
        # Value in bin j of dimension d: -5 + (j + rng.random()) * (10.0 / n_init)
        # Actually, to be more robust, we can just use the center of the bins or random within bins.
        # Let's use random within bins for diversity.
        init_points = []
        for i in range(n_init):
            x = np.zeros(dim)
            for d in range(dim):
                bin_idx = perm_matrix[d, i]
                # Range of bin: [ -5 + bin_idx * (10/n_init), -5 + (bin_idx+1) * (10/n_init) ]
                low = -5.0 + bin_idx * (10.0 / n_init)
                high = -5.0 + (bin_idx + 1) * (10.0 / n_init)
                x[d] = rng.uniform(low, high)
            init_points.append(x)
        
        # Evaluate initial points
        for x in init_points:
            fx = timed_call(x)
            if fx is None:
                break

        if best_x is None:
            # Fallback
            x = np.zeros(dim)
            fx = func(x)
            calls_made += 1
            return x, fx

        # Phase 2: Nelder-Mead
        # Initialize simplex
        # Vertices: best_x, and best_x + delta * e_i for i in 1..dim
        # Choose delta based on the scale of the problem. 
        # Since the box is [-5, 5], a reasonable initial step is 0.5 or 1.0.
        # Let's use 0.5 to stay within bounds initially, but clip is handled by func.
        delta = 0.5
        
        simplex = [best_x.copy()]
        for i in range(dim):
            v = best_x.copy()
            v[i] += delta
            simplex.append(v)
        
        # Evaluate simplex vertices
        simplex_f = [best_f]
        for i in range(1, dim + 1):
            fx = timed_call(simplex[i])
            if fx is None:
                break
            simplex_f.append(fx)
        
        # If we ran out of budget during simplex init, return best
        if len(simplex_f) < dim + 1:
            return best_x, best_f

        # Nelder-Mead Parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrinkage
        
        # Main Loop
        while calls_made < self.budget:
            if time.time() > deadline - 2:
                break
                
            # Sort simplex by function value
            # indices = np.argsort(simplex_f)
            # This is O(N log N), N=dim+1=6, so very fast.
            # But to avoid overhead, we can just find min and max.
            # However, standard NM requires the simplex to be ordered.
            # Let's maintain order.
            
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Check convergence
            # If the difference between best and worst is small, or step size is small
            f_best = simplex_f[0]
            f_worst = simplex_f[-1]
            # Relative tolerance
            if abs(f_worst - f_best) < 1e-8 * (1 + abs(f_best)):
                break
            
            # Compute centroid of all but the worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = timed_call(x_r)
            if f_r is None:
                break
            
            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5, 5)
                f_e = timed_call(x_e)
                if f_e is None:
                    break
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Reflection is good, accept it
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                # Check if we should do outside or inside contraction
                if f_r < simplex_f[-1]:
                    # Outside contraction
                    x_c = centroid + 0.5 * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - 0.5 * (simplex[-1] - centroid)
                
                x_c = np.clip(x_c, -5, 5)
                f_c = timed_call(x_c)
                if f_c is None:
                    break
                
                if f_c < simplex_f[-1]:
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrinkage
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_s = timed_call(simplex[i])
                        if f_s is None:
                            break
                        simplex_f[i] = f_s
                    if f_s is None:
                        break

        return best_x, best_f

# EVOLVE-BLOCK-END
