"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid Random Search + Nelder-Mead Optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s limit
        
        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        best_x = None
        best_f = float('inf')
        
        calls = 0
        rng = np.random.default_rng(42)
        
        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                return func(x)
            except:
                return None

        # Phase 1: Global Random Search
        # Use 20% of budget or at least 50 calls for global search
        n_random = min(100, max(50, budget // 5))
        
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = safe_call(x)
            if f is None:
                break
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        if best_x is None:
            # Fallback if no calls made
            x = np.zeros(dim)
            f = safe_call(x)
            if f is not None:
                best_x = x
                best_f = f
            return best_x, best_f

        # Phase 2: Local Refinement with Nelder-Mead
        # Initialize simplex around the best point found
        center = best_x.copy()
        simplex = []
        f_values = []
        
        # First vertex at center
        simplex.append(center.copy())
        f_values.append(best_f)
        
        # Other vertices along axes with initial step size
        # Use a smaller step size for refinement
        step = 0.5 
        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            xi = center.copy()
            xi[i] += step
            fi = safe_call(xi)
            if fi is None:
                break
            simplex.append(xi)
            f_values.append(fi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
                
        if calls >= budget or len(simplex) < dim + 1:
            return best_x, best_f
            
        simplex = np.array(simplex)
        f_values = np.array(f_values)
        
        # Main NM Loop
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Sort simplex by function value
            idx = np.argsort(f_values)
            simplex = simplex[idx]
            f_values = f_values[idx]
            
            # Update global best
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = simplex[0].copy()
                
            # Check for convergence/stagnation
            spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            if np.max(spread) < 1e-6 or np.max(f_values) - np.min(f_values) < 1e-10:
                # Restart with a smaller step or break
                # If we are stuck, just break to save time for other potential strategies?
                # Or restart with random perturbation?
                # For now, break to avoid wasting calls on a converged simplex
                break

            # Standard Nelder-Mead steps
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = safe_call(xr)
            if fr is None:
                break
            
            if fr < f_values[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
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
                xc = np.clip(xc, -5, 5)
                fc = safe_call(xc)
                if fc is None:
                    break
                if fc <= min(fr, f_values[-1]):
                    simplex[-1] = xc
                    f_values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fi = safe_call(simplex[i])
                        if fi is None:
                            break
                        f_values[i] = fi
                    if fi is None:
                        break

        return best_x, best_f

# EVOLVE-BLOCK-END
