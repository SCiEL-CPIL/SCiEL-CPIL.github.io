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
        deadline = time.time() + 290.0 
        
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
        # Use a small fraction of the budget to find a good starting region.
        n_random = 20
        
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

        # Phase 2: Nelder-Mead Local Refinement
        # Start from the best point found in Phase 1.
        x_start = best_x.copy()
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initialize simplex around x_start
        # We want a simplex that is large enough to see the ridge direction
        # but not so large it jumps out of the basin.
        # Initial step size: 1.0 is a reasonable choice for [-5,5] domain.
        step_size = 1.0
        
        simplex = [x_start]
        f_values = [best_f]
        
        # Create other vertices along axes
        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            xi = x_start.copy()
            xi[i] += step_size
            fi = safe_call(xi)
            if fi is None:
                break
            simplex.append(xi)
            f_values.append(fi)
            
            # Also try negative direction if budget allows? 
            # Standard NM uses positive offsets from center. 
            # To be more robust, we could initialize with +/- steps, 
            # but that doubles the init cost. 
            # Let's stick to standard NM init for simplicity and speed.
        
        if len(simplex) < dim + 1:
            # If we couldn't initialize fully, just return best so far
            return best_x, best_f

        simplex = np.array(simplex)
        f_values = np.array(f_values)
        
        # Main Nelder-Mead Loop
        stall_count = 0
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Sort simplex by function value (best first)
            idx = np.argsort(f_values)
            simplex = simplex[idx]
            f_values = f_values[idx]
            
            # Update global best
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = simplex[0].copy()
                stall_count = 0
            else:
                stall_count += 1

            # Check for convergence or stalling
            spread_x = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            spread_f = np.max(f_values) - np.min(f_values)
            
            # If simplex is very small OR we are stalling (no improvement for 50 iters)
            if (np.max(spread_x) < 1e-4 or spread_f < 1e-8) or (stall_count > 50):
                
                # If we have enough budget to restart
                if calls < budget - 2 * (dim + 1):
                    # Restart with a larger step to escape local minima or flat areas
                    center = simplex[0].copy()
                    new_simplex = [center]
                    new_f_values = [f_values[0]]
                    
                    # Use a step size that is larger than the current spread
                    current_max_spread = np.max(spread_x)
                    step = max(0.5, current_max_spread * 0.5)
                    
                    for i in range(dim):
                        if calls >= budget or time.time() > deadline:
                            break
                        xi = center.copy()
                        xi[i] += step
                        fi = safe_call(xi)
                        if fi is None:
                            break
                        new_simplex.append(xi)
                        new_f_values.append(fi)
                    
                    if len(new_simplex) == dim + 1:
                        simplex = np.array(new_simplex)
                        f_values = np.array(new_f_values)
                        stall_count = 0
                        continue
                else:
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
