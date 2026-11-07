import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or not time_ok():
                return np.inf
            calls += 1
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Multi-start Nelder-Mead parameters
        n_starts = 10
        max_iter_per_start = 45  # Each NM simplex has dim+1=6 points, so ~45 evals per start
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrink

        # Generate deterministic spread points in [-4, 4]^5
        # Use a grid-like distribution with some randomness for robustness
        rng = np.random.default_rng(123)
        
        start_points = []
        # Create 10 distinct deterministic starting points
        # Use a combination of center, corners, and random points
        base_points = [
            np.zeros(dim),                                    # Center
            np.array([4.0]*dim),                              # Corner +
            np.array([-4.0]*dim),                             # Corner -
            np.array([4.0, -4.0, 4.0, -4.0, 4.0]),           # Alternating
            np.array([-4.0, 4.0, -4.0, 4.0, -4.0]),          # Alternating
            np.array([0.0, 4.0, 0.0, 4.0, 0.0]),             # Half positive
            np.array([4.0, 0.0, 4.0, 0.0, 4.0]),             # Half positive
            np.array([-2.0, 2.0, -2.0, 2.0, -2.0]),          # Medium scale
            np.array([2.0, -2.0, 2.0, -2.0, 2.0]),           # Medium scale
        ]
        
        # Add one random point for diversity
        start_points.append(rng.uniform(-3.0, 3.0, size=dim))
        
        # Shuffle to avoid bias in ordering
        rng.shuffle(start_points)

        for s in range(n_starts):
            if calls >= budget or not time_ok():
                break
            
            # Initialize simplex around the start point
            start = start_points[s].copy()
            f_start = eval_func(start)
            if not time_ok() or calls >= budget:
                break
            
            # Build simplex: start point + dim perturbed points
            simplex = [start.copy()]
            simplex_f = [f_start]
            
            step_size = 0.5
            for j in range(dim):
                x_j = start.copy()
                x_j[j] += step_size
                x_j = np.clip(x_j, -5.0, 5.0)
                f_j = eval_func(x_j)
                if not time_ok() or calls >= budget:
                    break
                simplex.append(x_j)
                simplex_f.append(f_j)
            
            if not time_ok() or calls >= budget:
                break
            
            # Nelder-Mead iterations
            no_improve = 0
            max_no_improve = 8
            
            for it in range(max_iter_per_start):
                if calls >= budget or not time_ok():
                    break
                
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                f_best = simplex_f[0]
                f_worst = simplex_f[-1]
                f_second_worst = simplex_f[-2]
                
                # Check for convergence
                if f_worst - f_best < 1e-10:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = eval_func(x_r)
                
                if not time_ok() or calls >= budget:
                    break
                
                if f_r < f_second_worst:
                    # Expansion
                    if f_r < f_best:
                        x_e = centroid + gamma * (centroid - simplex[-1])
                        x_e = np.clip(x_e, -5.0, 5.0)
                        f_e = eval_func(x_e)
                        
                        if not time_ok() or calls >= budget:
                            break
                        
                        if f_e < f_r:
                            simplex[-1] = x_e
                            simplex_f[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    no_improve = 0
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = eval_func(x_c)
                    
                    if not time_ok() or calls >= budget:
                        break
                    
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                        no_improve = 0
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_shrink = eval_func(simplex[i])
                            
                            if not time_ok() or calls >= budget:
                                break
                            
                            simplex_f[i] = f_shrink
                        
                        no_improve += 1
                        if no_improve >= max_no_improve:
                            break
                
                # Update global best
                if f_best < best_f:
                    best_f = f_best
                    best_x = simplex[0].copy()

        return best_x, best_f
