"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 such that ||F(x)|| <= 1e-8.
    Uses a hybrid approach:
    1. Try Newton's method with adaptive step size (damping).
    2. If Newton fails or is slow, fall back to a derivative-free
       trust-region-like approach on ||F(x)||^2.
    3. Time guard to stay within budget.
    """
    x = np.array(x0, dtype=float).copy()
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    def elapsed():
        return time.time() - start_time
    
    def is_valid(x):
        return (x is not None and x.shape == (2,) and 
                np.all(np.isfinite(x)))
    
    # First, check if x0 is already a root
    try:
        Fx = F(x)
        if np.linalg.norm(Fx) <= 1e-8:
            return x
    except Exception:
        pass
    
    best_x = x.copy()
    best_norm = np.inf
    
    try:
        Fx = F(x)
        if np.linalg.norm(Fx) < best_norm:
            best_norm = np.linalg.norm(Fx)
            best_x = x.copy()
    except Exception:
        pass
    
    # Phase 1: Damped Newton's method with finite-difference Jacobian
    # Use adaptive perturbation for Jacobian estimation
    for iteration in range(80):
        if elapsed() > deadline:
            break
        
        try:
            Fx = F(x)
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x.copy()
            if norm_Fx <= 1e-8:
                return x
        except Exception:
            break
        
        # Estimate Jacobian with central differences, adaptive step
        h = 1e-6
        J = np.zeros((2, 2))
        try:
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_plus = F(x + e)
                Fx_minus = F(x - e)
                J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
        except Exception:
            # Try forward differences
            try:
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h
                    Fx_plus = F(x + e)
                    J[:, j] = (Fx_plus - Fx) / h
            except Exception:
                break
        
        # Solve Newton step
        try:
            step = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Try pseudo-inverse
            try:
                step = np.linalg.pinv(J) @ Fx
            except Exception:
                break
        
        # Damped step: try full step first, then reduce
        alpha = 1.0
        converged = False
        for _ in range(20):
            x_trial = x - alpha * step
            if not is_valid(x_trial):
                alpha *= 0.5
                continue
            try:
                Fx_trial = F(x_trial)
                norm_trial = np.linalg.norm(Fx_trial)
            except Exception:
                alpha *= 0.5
                continue
            
            if norm_trial < best_norm:
                best_norm = norm_trial
                best_x = x_trial.copy()
            
            # Accept if we made progress or are close enough
            if norm_trial <= norm_Fx * 0.5 or norm_trial <= 1e-8:
                x = x_trial
                if norm_trial <= 1e-8:
                    return x
                converged = True
                break
            else:
                alpha *= 0.5
        
        if not converged:
            # If we couldn't make progress with damped Newton, try a different approach
            break
        
        if elapsed() > deadline:
            break
    
    # Phase 2: If Newton didn't converge, try a simple derivative-free approach
    # Use Nelder-Mead style or a simple coordinate descent on ||F(x)||^2
    if elapsed() < deadline:
        def objective(x_trial):
            try:
                Fx_trial = F(x_trial)
                return float(np.dot(Fx_trial, Fx_trial))
            except Exception:
                return 1e100
        
        # Start from best_x
        x_nm = best_x.copy()
        obj_val = objective(x_nm)
        
        # Nelder-Mead simplex
        simplex = [x_nm.copy()]
        for i in range(2):
            simplex.append(x_nm + 0.1 * np.eye(2)[i])
        
        for _ in range(200):
            if elapsed() > deadline:
                break
            
            # Evaluate objective at all simplex vertices
            objs = [objective(v) for v in simplex]
            
            # Sort
            order = np.argsort(objs)
            simplex = [simplex[i] for i in order]
            objs = [objs[i] for i in order]
            
            if objs[0] <= 1e-16:  # ||F||^2 <= (1e-8)^2
                return simplex[0]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_reflect = centroid + (centroid - simplex[-1])
            obj_reflect = objective(x_reflect)
            
            if obj_reflect < objs[0]:
                # Expansion
                x_expand = centroid + 2 * (centroid - simplex[-1])
                obj_expand = objective(x_expand)
                if obj_expand < obj_reflect:
                    simplex[-1] = x_expand
                else:
                    simplex[-1] = x_reflect
            elif obj_reflect < objs[-2]:
                simplex[-1] = x_reflect
            else:
                # Contraction
                x_contract = centroid + 0.5 * (simplex[-1] - centroid)
                obj_contract = objective(x_contract)
                if obj_contract < objs[-1]:
                    simplex[-1] = x_contract
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
    
    # Return the best result found
    if is_valid(best_x):
        return best_x
    return x
# EVOLVE-BLOCK-END
