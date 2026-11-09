import numpy as np
import time

class Optimizer:
    """
    Optimizer using multi-start Nelder-Mead with adaptive step sizes and coordinate descent polishing.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320.0  # safety margin before 330s kill
        dim = self.dim
        total_budget = self.budget
        used = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def elapsed():
            return time.time() - t0

        def check_deadline():
            return elapsed() > deadline

        def evaluate(x):
            nonlocal used, best_f, best_x
            x_clip = np.clip(x, -5, 5)
            f_val = func(x_clip)
            used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clip.copy()
            return f_val

        # Nelder-Mead implementation
        def nelder_mead(x0, max_iter=100):
            nonlocal used, best_f, best_x
            if check_deadline() or used >= total_budget:
                return x0
            
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.5 if xi[i] == 0 else 0.1 * xi[i]
                simplex.append(xi)
            
            # Evaluate initial simplex
            f_values = []
            for i, s in enumerate(simplex):
                if check_deadline() or used >= total_budget:
                    break
                f_values.append(evaluate(s))
            
            if len(f_values) < dim + 1:
                return x0
            
            # Sort simplex
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]
            
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            for _ in range(max_iter):
                if check_deadline() or used >= total_budget:
                    break
                
                # Calculate centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = evaluate(x_r)
                
                if f_values[0] <= f_r < f_values[-2]:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                elif f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_values[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    
                    f_c = evaluate(x_c)
                    if f_c < min(f_r, f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if check_deadline() or used >= total_budget:
                                break
                            f_values[i] = evaluate(simplex[i])
                        if check_deadline() or used >= total_budget:
                            break
                
                # Sort simplex
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                # Convergence check
                if np.std(f_values) < 1e-10:
                    break
            
            return simplex[0]

        # Coordinate descent polishing
        def coordinate_descent(x0, max_iter=50):
            nonlocal used, best_f, best_x
            if check_deadline() or used >= total_budget:
                return x0
            
            x = x0.copy()
            f = best_f
            step_sizes = np.ones(dim) * 0.1
            
            for _ in range(max_iter):
                if check_deadline() or used >= total_budget:
                    break
                
                improved = False
                for i in range(dim):
                    if check_deadline() or used >= total_budget:
                        break
                    
                    # Try positive direction
                    x_plus = x.copy()
                    x_plus[i] += step_sizes[i]
                    x_plus = np.clip(x_plus, -5, 5)
                    f_plus = evaluate(x_plus)
                    
                    if f_plus < f:
                        x = x_plus
                        f = f_plus
                        improved = True
                    else:
                        # Try negative direction
                        x_minus = x.copy()
                        x_minus[i] -= step_sizes[i]
                        x_minus = np.clip(x_minus, -5, 5)
                        f_minus = evaluate(x_minus)
                        
                        if f_minus < f:
                            x = x_minus
                            f = f_minus
                            improved = True
                        else:
                            # Shrink step size
                            step_sizes[i] *= 0.5
                            if step_sizes[i] < 1e-6:
                                step_sizes[i] = 1e-6
                
                if not improved:
                    break
            
            return x

        # Multi-start strategy
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        start_points = [np.zeros(dim)]
        
        # Axis-aligned points
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 3.0
                start_points.append(p)
        
        # Random points
        num_random_starts = 10
        for _ in range(num_random_starts):
            p = rng.uniform(-4, 4, dim)
            start_points.append(p)
        
        # Allocate budget: 20% for each start, but cap at 50 evaluations per start
        budget_per_start = max(10, min(50, total_budget // len(start_points)))
        
        for start in start_points:
            if check_deadline() or used >= total_budget:
                break
            
            # Run Nelder-Mead from this start
            x_nm = nelder_mead(start, max_iter=budget_per_start)
            
            # Polish with coordinate descent if budget remains
            if not check_deadline() and used < total_budget:
                x_cd = coordinate_descent(x_nm, max_iter=20)
        
        return best_x, best_f
