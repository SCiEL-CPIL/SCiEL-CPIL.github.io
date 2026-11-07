import numpy as np
import time

def find_root_system(F, x0):
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = float('inf')
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    calls = 0
    
    def safe_F(x):
        nonlocal calls
        if time.time() > deadline:
            raise TimeoutError
        if calls >= 495:
            raise RuntimeError("Call limit exceeded")
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return None
            calls += 1
            return Fx
        except Exception:
            return None

    # 1. Initial Eval
    Fx = safe_F(x)
    if Fx is None: return best_x
    norm = np.linalg.norm(Fx)
    if norm < best_norm:
        best_norm = norm
        best_x = x.copy()
    if norm <= 1e-8: return best_x

    # 2. Broyden Phase
    # Compute Jacobian
    h = 1e-6
    J = np.zeros((2, 2))
    jacobian_valid = True
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_j = safe_F(x + e)
        if Fx_j is None:
            jacobian_valid = False
            break
        J[:, j] = (Fx_j - Fx) / h
    
    if not jacobian_valid:
        J = np.eye(2)
        
    max_broyden_iter = 100
    for i in range(max_broyden_iter):
        if time.time() > deadline: break
        if calls >= 490: break
        
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
            
        # Line Search
        alpha = 1.0
        step_accepted = False
        for _ in range(10): # Max 10 line search steps
            x_try = x + alpha * dx
            Fx_try = safe_F(x_try)
            if Fx_try is None:
                alpha *= 0.5
                continue
                
            norm_try = np.linalg.norm(Fx_try)
            if norm_try < best_norm:
                best_norm = norm_try
                best_x = x_try.copy()
                
            if norm_try < norm * 0.9: # Sufficient decrease
                # Update Broyden
                s = alpha * dx
                y = Fx_try - Fx
                J_s = J @ s
                diff = y - J_s
                denom = s @ s
                if denom > 1e-15:
                    J = J + np.outer(diff, s) / denom
                else:
                    J = np.eye(2) # Reset if step is tiny
                
                x = x_try
                Fx = Fx_try
                norm = norm_try
                step_accepted = True
                break
            else:
                alpha *= 0.5
                
        if not step_accepted:
            break
        if norm <= 1e-8:
            return best_x
            
    # 3. Nelder-Mead Phase (if not converged)
    if norm > 1e-8:
        # Setup Nelder-Mead simplex
        # Start from best_x
        x0_nm = best_x.copy()
        
        # Create simplex
        # Points: x0, x0 + delta*e1, x0 + delta*e2
        delta = 0.1
        p0 = x0_nm
        p1 = x0_nm + np.array([delta, 0])
        p2 = x0_nm + np.array([0, delta])
        
        # Evaluate initial simplex
        f0 = safe_F(p0)
        f1 = safe_F(p1)
        f2 = safe_F(p2)
        
        if f0 is None or f1 is None or f2 is None:
            return best_x
            
        norm0 = np.linalg.norm(f0)
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        
        # Update best
        if norm0 < best_norm: best_norm = norm0; best_x = p0.copy()
        if norm1 < best_norm: best_norm = norm1; best_x = p1.copy()
        if norm2 < best_norm: best_norm = norm2; best_x = p2.copy()
        
        # Sort by norm
        # We need to keep track of points and their norms
        points = [p0, p1, p2]
        norms = [norm0, norm1, norm2]
        
        # Sort indices
        order = np.argsort(norms)
        points = [points[i] for i in order]
        norms = [norms[i] for i in order]
        
        max_nm_iter = 300
        for _ in range(max_nm_iter):
            if time.time() > deadline: break
            if calls >= 490: break
            
            # Current worst is points[2], best is points[0]
            # Centroid of best 2
            centroid = (points[0] + points[1]) / 2.0
            
            # Reflection
            alpha_nm = 1.0
            p_r = centroid + alpha_nm * (centroid - points[2])
            f_r = safe_F(p_r)
            if f_r is None:
                break
            norm_r = np.linalg.norm(f_r)
            if norm_r < best_norm:
                best_norm = norm_r
                best_x = p_r.copy()
                
            if norm_r < norms[1]:
                # Reflection is better than 2nd best
                # Check if it's better than best (Expansion)
                if norm_r < norms[0]:
                    gamma = 2.0
                    p_e = centroid + gamma * (centroid - points[2])
                    f_e = safe_F(p_e)
                    if f_e is None:
                        break
                    norm_e = np.linalg.norm(f_e)
                    if norm_e < best_norm:
                        best_norm = norm_e
                        best_x = p_e.copy()
                        
                    if norm_e < norm_r:
                        points[2] = p_e
                        norms[2] = norm_e
                    else:
                        points[2] = p_r
                        norms[2] = norm_r
                else:
                    points[2] = p_r
                    norms[2] = norm_r
            else:
                # Reflection is worse than 2nd best
                # Contraction
                if norm_r < norms[2]:
                    # Inside contraction
                    rho = 0.5
                    p_c = centroid + rho * (centroid - points[2])
                    f_c = safe_F(p_c)
                    if f_c is None:
                        break
                    norm_c = np.linalg.norm(f_c)
                    if norm_c < best_norm:
                        best_norm = norm_c
                        best_x = p_c.copy()
                        
                    if norm_c < norm_r:
                        points[2] = p_c
                        norms[2] = norm_c
                    else:
                        # Shrink
                        for i in range(1, 3):
                            points[i] = points[0] + 0.5 * (points[i] - points[0])
                            f_i = safe_F(points[i])
                            if f_i is None:
                                return best_x
                            norms[i] = np.linalg.norm(f_i)
                            if norms[i] < best_norm:
                                best_norm = norms[i]
                                best_x = points[i].copy()
                else:
                    # Shrink
                    for i in range(1, 3):
                        points[i] = points[0] + 0.5 * (points[i] - points[0])
                        f_i = safe_F(points[i])
                        if f_i is None:
                            return best_x
                        norms[i] = np.linalg.norm(f_i)
                        if norms[i] < best_norm:
                            best_norm = norms[i]
                            best_x = points[i].copy()
                            
            # Re-sort
            order = np.argsort(norms)
            points = [points[i] for i in order]
            norms = [norms[i] for i in order]
            
            if norms[0] <= 1e-8:
                return best_x
                
    return best_x
