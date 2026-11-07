"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x)=0 using the Nelder-Mead simplex method.
    This is a derivative-free optimization approach applied to minimize ||F(x)||^2.
    """
    x0 = np.array(x0, dtype=float)
    
    # Objective function: minimize ||F(x)||^2
    def objective(x):
        fx = F(x)
        if not np.all(np.isfinite(fx)):
            return np.inf
        return np.dot(fx, fx)
    
    # Initial simplex
    # Start with x0 and two other points close to it
    delta = 0.1
    simplex = np.array([
        x0,
        x0 + np.array([delta, 0.0]),
        x0 + np.array([0.0, delta])
    ])
    
    # Evaluate initial simplex
    f_vals = np.array([objective(simplex[i]) for i in range(3)])
    
    # If we already found a root
    if np.min(f_vals) < (1e-8)**2:
        idx = np.argmin(f_vals)
        return simplex[idx]
    
    # Nelder-Mead parameters
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    max_iter = 200
    tol = 1e-12  # Tight tolerance for function value
    
    for _ in range(max_iter):
        # Sort simplex by function values
        order = np.argsort(f_vals)
        simplex = simplex[order]
        f_vals = f_vals[order]
        
        x0_sm = simplex[0]
        x1_sm = simplex[1]
        x2_sm = simplex[2]
        f0 = f_vals[0]
        f1 = f_vals[1]
        f2 = f_vals[2]
        
        # Check convergence
        if f0 < (1e-8)**2:
            return x0_sm
        
        # Check simplex size
        if np.linalg.norm(x2_sm - x0_sm) < 1e-10 and np.linalg.norm(x1_sm - x0_sm) < 1e-10:
            break
            
        centroid = (x0_sm + x1_sm) / 2.0
        
        # Reflection
        xr = centroid + alpha * (centroid - x2_sm)
        fr = objective(xr)
        
        if fr < f0:
            # Expansion
            xe = centroid + gamma * (centroid - x2_sm)
            fe = objective(xe)
            if fe < fr:
                simplex[2] = xe
                f_vals[2] = fe
            else:
                simplex[2] = xr
                f_vals[2] = fr
        elif fr < f1:
            # Reflection is good
            simplex[2] = xr
            f_vals[2] = fr
        else:
            # Contraction
            if fr < f2:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
                fc = objective(xc)
                if fc < fr:
                    simplex[2] = xc
                    f_vals[2] = fc
                else:
                    # Shrink
                    for i in range(1, 3):
                        simplex[i] = x0_sm + sigma * (simplex[i] - x0_sm)
                        f_vals[i] = objective(simplex[i])
            else:
                # Inside contraction
                xc = centroid - rho * (centroid - x2_sm)
                fc = objective(xc)
                if fc < f2:
                    simplex[2] = xc
                    f_vals[2] = fc
                else:
                    # Shrink
                    for i in range(1, 3):
                        simplex[i] = x0_sm + sigma * (simplex[i] - x0_sm)
                        f_vals[i] = objective(simplex[i])
    
    # Return the best point found
    order = np.argsort(f_vals)
    best_x = simplex[order[0]]
    
    # Final check
    fb = F(best_x)
    if np.linalg.norm(fb) <= 1e-8:
        return best_x
    
    # If failed, try a few random restarts or just return best
    # Given the constraint, we return the best found
    return best_x
# EVOLVE-BLOCK-END
