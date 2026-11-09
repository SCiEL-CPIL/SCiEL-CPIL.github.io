"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 such that ||F(x)|| <= 1e-8 using a hybrid
    derivative-free Levenberg-Marquardt with rank-1 Jacobian updates
    and adaptive damping, combined with Nelder-Mead fallback.
    """
    x0 = np.asarray(x0, dtype=float)
    deadline = time.time() + 115.0  # safety margin before 120s limit
    
    def norm_F(x):
        Fx = F(x)
        return np.linalg.norm(Fx)
    
    def check_converged(x, Fx):
        return np.linalg.norm(Fx) <= 1e-8
    
    # Phase 1: LM with rank-1 Jacobian update
    x = x0.copy()
    Fx = F(x)
    fx_norm = np.linalg.norm(Fx)
    
    if check_converged(x, Fx):
        return x
    
    # Initial Jacobian via finite differences
    h = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (F(x + e) - Fx) / h
    
    # Adaptive damping
    lam = 0.1
    best_x = x.copy()
    best_norm = fx_norm
    best_Fx = Fx.copy()
    
    for it in range(300):
        if time.time() > deadline:
            break
        if not np.all(np.isfinite(x)):
            break
        
        # Check convergence
        if fx_norm <= 1e-8:
            return x
        
        # Update best
        if fx_norm < best_norm:
            best_norm = fx_norm
            best_x = x.copy()
            best_Fx = Fx.copy()
        
        # Levenberg-Marquardt step
        JtJ = J.T @ J
        JtF = J.T @ Fx
        A = JtJ + lam * np.eye(2)
        try:
            step = np.linalg.solve(A, JtF)
        except np.linalg.LinAlgError:
            lam *= 10
            continue
        
        # Try step
        x_new = x + step
        Fx_new = F(x_new)
        new_norm = np.linalg.norm(Fx_new)
        
        # Adaptive damping update
        if new_norm < fx_norm:
            # Accept step, decrease lambda
            x = x_new
            Fx = Fx_new
            fx_norm = new_norm
            lam = max(lam * 0.5, 1e-10)
            
            # Rank-1 Jacobian update
            J[:, 1] = (F(x + h * np.array([0, 1])) - Fx) / h
            # Actually recompute second column periodically
            if it % 10 == 0:
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h
                    J[:, j] = (F(x + e) - Fx) / h
        else:
            # Reject step, increase lambda
            lam = min(lam * 10.0, 1e10)
    
    # If LM converged, return
    if fx_norm <= 1e-8:
        return x
    if best_norm <= 1e-8:
        return best_x
    
    # Phase 2: Nelder-Mead fallback on ||F(x)||^2
    def objective(x):
        Fx = F(x)
        return float(np.dot(Fx, Fx))
    
    # Start from best point found
    x_nm = best_x.copy()
    f0 = objective(x_nm)
    
    if f0 <= 1e-16:
        return x_nm
    
    # Nelder-Mead simplex
    eps = 1e-5
    v = np.eye(2) * eps
    simplex = [x_nm, x_nm + v[0], x_nm + v[1]]
    fvals = [objective(s) for s in simplex]
    
    for nm_it in range(2000):
        if time.time() > deadline:
            break
        
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if fvals[0] <= 1e-16:
            return simplex[0]
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        f_worst = fvals[-1]
        
        # Reflection
        alpha = 1.0
        x_r = centroid + alpha * (centroid - worst)
        f_r = objective(x_r)
        
        if f_r < fvals[0]:
            # Expansion
            gamma = 2.0
            x_e = centroid + gamma * (centroid - worst)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            # Contraction
            rho = 0.5
            x_c = centroid + rho * (worst - centroid)
            f_c = objective(x_c)
            if f_c < f_worst:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, 3):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
        
        # Check if we found a root
        if fvals[0] <= 1e-16:
            return simplex[0]
    
    # Return best result
    if best_norm <= fx_norm:
        return best_x
    return x
# EVOLVE-BLOCK-END
