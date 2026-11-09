"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead Simplex Optimizer with Coordinate Descent Polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                return np.inf
            calls_used += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead(x0, max_iter):
            """
            Deterministic Nelder-Mead Simplex Method.
            """
            nonlocal calls_used, deadline, best_x, best_f
            
            # Initialize simplex
            simplex = [x0.copy()]
            f_vals = [call_func(x0)]
            
            # Create other vertices of the simplex
            for i in range(dim):
                xi = x0.copy()
                if f_vals[0] != 0:
                    step = 0.05 * abs(f_vals[0])
                else:
                    step = 0.05
                xi[i] += step
                simplex.append(xi)
                f_vals.append(call_func(xi))
            
            # Sort simplex by function value
            order = np.argsort(f_vals)
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]
            
            alpha = 1.0    # Reflection
            gamma = 2.0    # Expansion
            rho = 0.5      # Contraction
            sigma = 0.5    # Shrinkage
            
            for _ in range(max_iter):
                if time.time() > deadline or calls_used >= budget:
                    break
                
                # 1. Reflection
                centroid = np.mean(simplex[:-1], axis=0)
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = call_func(x_ref)
                
                if f_ref < f_vals[0]:
                    # 2. Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = call_func(x_exp)
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_vals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_vals[-1] = f_ref
                elif f_ref < f_vals[-2]:
                    # 3. Just Reflection
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
                else:
                    # 4. Contraction
                    if f_ref < f_vals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                        f_con = call_func(x_con)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[-1] - centroid)
                        f_con = call_func(x_con)
                    
                    if f_con < min(f_ref, f_vals[-1]):
                        simplex[-1] = x_con
                        f_vals[-1] = f_con
                    else:
                        # 5. Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_vals[i] = call_func(simplex[i])
                
                # Re-sort
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]
                
                # Check convergence
                if np.std(f_vals) < 1e-10:
                    break
            
            return simplex[0], f_vals[0]

        def coordinate_descent(x0, max_iter):
            """
            Coordinate Descent Polish.
            """
            nonlocal calls_used, deadline, best_x, best_f
            
            x = x0.copy()
            f_x = call_func(x)
            step_sizes = np.ones(dim) * 0.1
            
            for _ in range(max_iter):
                if time.time() > deadline or calls_used >= budget:
                    break
                
                improved = False
                for i in range(dim):
                    # Try positive step
                    x_trial = x.copy()
                    x_trial[i] += step_sizes[i]
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = call_func(x_trial)
                    
                    if f_trial < f_x:
                        x = x_trial
                        f_x = f_trial
                        improved = True
                        continue
                    
                    # Try negative step
                    x_trial = x.copy()
                    x_trial[i] -= step_sizes[i]
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = call_func(x_trial)
                    
                    if f_trial < f_x:
                        x = x_trial
                        f_x = f_trial
                        improved = True
                    else:
                        # Reduce step size for this dimension
                        step_sizes[i] *= 0.5
                        if step_sizes[i] < 1e-6:
                            step_sizes[i] = 1e-6
                
                if not improved:
                    break
            
            return x, f_x

        # 1. Generate diverse initial points
        rng = np.random.default_rng(42)
        
        starts = []
        # Standard origin
        starts.append(np.zeros(dim))
        
        # Theoretical minimum location
        starts.append(np.ones(dim))
        
        # Random points
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, dim))
            
        # Diagonal points (often helpful in Rosenbrock)
        for val in [-2, 0, 2]:
            starts.append(np.full(dim, val))

        # 2. Evaluate starts and pick the best
        best_start_x = None
        best_start_f = np.inf
        
        for s in starts:
            if time.time() > deadline or calls_used >= budget:
                break
            s = np.clip(s, -5, 5)
            f_s = call_func(s)
            if f_s < best_start_f:
                best_start_f = f_s
                best_start_x = s.copy()
        
        if best_start_x is not None:
            # 3. Run Nelder-Mead from the best start
            remaining_calls = budget - calls_used
            # Nelder-Mead uses ~dim+1 calls per iteration. 
            # Let's allocate half the remaining budget to NM.
            max_iter_nm = int(remaining_calls * 0.5 / (dim + 1))
            if max_iter_nm > 0:
                x_nm, f_nm = nelder_mead(best_start_x, max_iter_nm)
                
                # 4. Run Coordinate Descent from the result of NM
                remaining_calls = budget - calls_used
                # CD uses ~2*dim calls per full sweep.
                max_iter_cd = int(remaining_calls * 0.8 / (2 * dim))
                if max_iter_cd > 0:
                    x_cd, f_cd = coordinate_descent(x_nm, max_iter_cd)
                    
                    # 5. If budget remains, try another start with NM
                    remaining_calls = budget - calls_used
                    if remaining_calls > 50:
                        # Pick a different start
                        alt_start = starts[0] if not np.allclose(starts[0], best_start_x) else starts[1]
                        alt_start = np.clip(alt_start, -5, 5)
                        max_iter_alt = int(remaining_calls * 0.8 / (dim + 1))
                        if max_iter_alt > 0:
                            x_alt, f_alt = nelder_mead(alt_start, max_iter_alt)
                            # Polish the alternative too if possible
                            remaining_calls = budget - calls_used
                            if remaining_calls > 20:
                                max_iter_cd_alt = int(remaining_calls * 0.8 / (2 * dim))
                                if max_iter_cd_alt > 0:
                                    coordinate_descent(x_alt, max_iter_cd_alt)

        return best_x, best_f
# EVOLVE-BLOCK-END
