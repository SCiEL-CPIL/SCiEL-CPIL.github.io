"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using analytical front sampling and local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        n_angle = dim - n_obj  # 9 for 12 vars, 3 objs

        # Phase 1: Sample points on the known DTLZ2 Pareto front
        # DTLZ2: f_i = (1+g) * cos(theta_1 * cos(theta_2) * ... * cos(theta_{i-1}) * sin(theta_i) * ...)
        # For 3 objectives, angles are theta_1, theta_2, theta_3 with constraints
        # Actually DTLZ2 with M objectives uses M-1 angles in [0, pi/2]
        # f_1 = (1+g)*cos(t1)*cos(t2)
        # f_2 = (1+g)*sin(t1)*cos(t2)
        # f_3 = (1+g)*sin(t2)
        # To get points on the front, set g=0 and choose angles uniformly
        # Then x[0:n_obj-1] are determined by angles, x[n_obj:] should minimize g
        
        # For DTLZ2, g(x) = sum((x_i - 0.5)^2) for i >= n_obj (or similar)
        # We need to find x such that the first n_obj-1 variables give the right angles
        # and the remaining variables minimize g
        
        # Actually, let's use a simpler approach:
        # Generate candidate solutions by sampling the Pareto front analytically
        # and then doing local search to minimize g while maintaining the front position
        
        # For DTLZ2 with M=3, the Pareto front is a quarter sphere
        # Points on the front: f_i = cos/sin combinations
        # The decision variables: x[0], x[1] determine the angles, x[2:] affect g
        
        # Let's generate diverse points on the Pareto front
        # and use local search to minimize g for each
        
        rng = np.random.default_rng(42)
        
        # Number of front points to sample
        n_front = min(50, self.budget // 4)
        used = 0
        
        # Store all evaluated points
        all_X = []
        all_F = []
        
        # Generate angles on the Pareto front
        # For M=3: t1, t2 in [0, pi/2]
        # f1 = cos(t1)*cos(t2), f2 = sin(t1)*cos(t2), f3 = sin(t2) (when g=0)
        
        for i in range(n_front):
            if time.time() > deadline:
                break
            
            # Sample angles
            t1 = rng.uniform(0, np.pi/2)
            t2 = rng.uniform(0, np.pi/2)
            
            # Construct x such that the objective values match the front
            # For DTLZ2, the mapping from x to f involves:
            # The first M-1 variables determine the angles
            # We need to find x[0], x[1] that give angles t1, t2
            # and x[2:] that minimize g
            
            # Simple approach: set x[0], x[1] to values that approximate the angles
            # and x[2:] to 0.5 (which minimizes g for DTLZ2)
            # Then do local search to improve
            
            x = np.zeros(dim)
            x[0] = t1 / (np.pi/2)  # Linear mapping approximation
            x[1] = t2 / (np.pi/2)
            x[2:] = 0.5  # Minimize g
            
            # Local search to minimize g while keeping x[0], x[1] roughly fixed
            # Use small steps in x[2:] and adjust x[0], x[1] if needed
            
            # Actually, let's just evaluate this point and do a few local improvements
            fx = problem(x)
            used += 1
            all_X.append(x.copy())
            all_F.append(fx.copy())
            
            # Local search: try small perturbations
            n_local = 3
            for j in range(n_local):
                if time.time() > deadline or used >= self.budget:
                    break
                
                # Perturb all variables slightly
                delta = 0.05 * (hi - lo) * rng.standard_normal(dim)
                x_new = np.clip(x + delta, lo, hi)
                
                fx_new = problem(x_new)
                used += 1
                
                # Accept if it improves the sum of objectives
                if np.sum(fx_new) < np.sum(fx):
                    x = x_new
                    fx = fx_new
                    all_X.append(x.copy())
                    all_F.append(fx.copy())
        
        # Phase 2: If budget remains, do random search with local refinement
        if used < self.budget:
            for _ in range(min(20, self.budget - used)):
                if time.time() > deadline:
                    break
                
                x = lo + (hi - lo) * rng.random(dim)
                fx = problem(x)
                used += 1
                all_X.append(x.copy())
                all_F.append(fx.copy())
                
                # Local search
                for _ in range(2):
                    if time.time() > deadline or used >= self.budget:
                        break
                    delta = 0.03 * (hi - lo) * rng.standard_normal(dim)
                    x_new = np.clip(x + delta, lo, hi)
                    fx_new = problem(x_new)
                    used += 1
                    if np.sum(fx_new) < np.sum(fx):
                        x = x_new
                        fx = fx_new
                        all_X.append(x.copy())
                        all_F.append(fx.copy())
        
        # Convert to arrays
        X = np.array(all_X)
        F = np.array(all_F)
        
        return X
# EVOLVE-BLOCK-END
