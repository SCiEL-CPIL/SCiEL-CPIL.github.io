"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Initialize simplex at the center (0,0,0,0,0)
        # For linear functions, the minimum is on the boundary.
        # Starting at 0 is safe and symmetric.
        x0 = np.zeros(dim)
        
        # Initial simplex: x0 and x0 + delta * e_i
        delta = 0.5  # Small initial step to stay within [-5, 5]
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += delta
            simplex.append(xi)
        
        # Evaluate initial simplex
        fvals = []
        calls_made = 0
        best_x = x0.copy()
        best_f = np.inf
        
        for i in range(dim + 1):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
            f = func(simplex[i])
            fvals.append(f)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = simplex[i].copy()
        
        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma_nm = 0.5 # Shrinkage
        
        # Main loop
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check for convergence
            # If the spread is very small, we might be stuck or converged.
            # For linear functions, we should hit a vertex or edge quickly.
            # Simple convergence check: max_f - min_f
            if fvals[-1] - fvals[0] < 1e-10:
                break
            
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            
            if time.time() > deadline or calls_made >= budget:
                break
            fr = func(xr)
            calls_made += 1
            if fr < best_f:
                best_f = fr
                best_x = xr.copy()
            
            # If reflection is better than the best, try expansion
            if fr < fvals[0]:
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                
                if time.time() > deadline or calls_made >= budget:
                    break
                fe = func(xe)
                calls_made += 1
                if fe < best_f:
                    best_f = fe
                    best_x = xe.copy()
                
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    
            # If reflection is better than second best, accept it
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
                
            # Otherwise, try contraction
            else:
                # Outside contraction
                xo = centroid + 0.5 * (xr - centroid)
                xo = np.clip(xo, -5, 5)
                
                if time.time() > deadline or calls_made >= budget:
                    break
                fo = func(xo)
                calls_made += 1
                if fo < best_f:
                    best_f = fo
                    best_x = xo.copy()
                
                if fo < fr:
                    simplex[-1] = xo
                    fvals[-1] = fo
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        if time.time() > deadline or calls_made >= budget:
                            break
                        fi = func(simplex[i])
                        fvals[i] = fi
                        calls_made += 1
                        if fi < best_f:
                            best_f = fi
                            best_x = simplex[i].copy()
                    if calls_made >= budget:
                        break
                        
        return best_x, best_f

# EVOLVE-BLOCK-END
