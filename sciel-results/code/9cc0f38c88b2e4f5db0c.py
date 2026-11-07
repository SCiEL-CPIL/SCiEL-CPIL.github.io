import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer: Global Random Search followed by Nelder-Mead Local Refinement.
    Designed for BBOB f13 (Sharp Ridge) in 5D with budget 500.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        # Safety margin: Evaluator kills at 330s. We target finishing well before.
        deadline = time.time() + 285.0 
        
        best_x = None
        best_f = float('inf')
        calls = 0
        rng = np.random.default_rng(1234) # Fixed seed for reproducibility within run
        
        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                f_val = func(x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global Random Search
        # Use more samples to better locate the basin in 5D
        n_random = 30
        
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            # Fallback if all random calls failed (unlikely)
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0

        # Phase 2: Nelder-Mead with restarts
        # We'll run NM from the best point, and if it stalls, restart from other good points
        
        def nelder_mead(x_start, f_start, max_iter=200):
            nonlocal calls, best_f, best_x, deadline
            
            if calls >= budget or time.time() > deadline:
                return best_x, best_f
            
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Initialize simplex around x_start
            step_size = 0.5  # Smaller initial step for better local refinement
            
            simplex = [x_start.copy()]
            f_values = [f_start]
            
            # Create other vertices along axes
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    return best_x, best_f
                xi = x_start.copy()
                xi[i] += step_size
                fi = safe_call(xi)
                if fi is None:
                    return best_x, best_f
                simplex.append(xi)
                f_values.append(fi)
            
            if len(simplex) < dim + 1:
                return best_x, best_f

            simplex = np.array(simplex)
            f_values = np.array(f_values)
            
            no_improve_count = 0
            patience = 15  # Number of iterations without improvement before restarting
            
            for iteration in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                    
                # Sort simplex by function value (best first)
                idx = np.argsort(f_values)
                simplex = simplex[idx]
                f_values = f_values[idx]
                
                # Update global best
                if f_values[0] < best_f:
                    best_f = f_values[0]
                    best_x = simplex[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Check for convergence
                spread_x = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                spread_f = np.max(f_values) - np.min(f_values)
                
                if np.max(spread_x) < 1e-5 or spread_f < 1e-8:
                    # Convergence reached - break to allow restart
                    break
                
                # Patience check
                if no_improve_count > patience:
                    break
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = safe_call(xr)
                if fr is None:
                    break
                
                if fr < f_values[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = safe_call(xe)
                    if fe is None:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        f_values[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_values[-1] = fr
                elif fr < f_values[-2]:
                    # Reflection accepted
                    simplex[-1] = xr
                    f_values[-1] = fr
                else:
                    # Contraction
                    if fr < f_values[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = safe_call(xc)
                    if fc is None:
                        break
                    if fc <= min(fr, f_values[-1]):
                        simplex[-1] = xc
                        f_values[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if calls >= budget or time.time() > deadline:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fi = safe_call(simplex[i])
                            if fi is None:
                                break
                            f_values[i] = fi
                        if fi is None:
                            break
            
            return best_x, best_f

        # Run NM from best point
        nelder_mead(best_x.copy(), best_f)
        
        # If we have budget left, try a few more restarts from slightly perturbed best points
        if calls < budget - 10 and time.time() < deadline:
            for restart in range(3):
                if calls >= budget - 5 or time.time() > deadline:
                    break
                
                # Perturb the best point slightly
                perturbation = rng.normal(0, 0.1, size=dim)
                x_restart = np.clip(best_x + perturbation, -5.0, 5.0)
                
                f_restart = safe_call(x_restart)
                if f_restart is None:
                    break
                
                if f_restart < best_f:
                    best_f = f_restart
                    best_x = x_restart.copy()
                
                # Run NM from this point
                nelder_mead(x_restart.copy(), f_restart, max_iter=50)
        
        return best_x, best_f
