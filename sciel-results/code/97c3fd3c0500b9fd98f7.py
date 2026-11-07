"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    x = np.array(x0, dtype=float)
    Fx = F(x)
    norm_F = np.linalg.norm(Fx)
    
    if norm_F <= 1e-8:
        return x
    
    best_x = x.copy()
    best_norm = norm_F
    
    def newton_from(x_start, F_start, max_iter=30, h=1e-5):
        """Damped Newton with line search. Returns (x, norm, calls_used)."""
        nonlocal best_x, best_norm
        x_curr = x_start.copy()
        Fx_curr = F_start.copy()
        norm_curr = np.linalg.norm(Fx_curr)
        calls = 1  # F_start already counted
        
        if norm_curr <= 1e-8:
            return x_curr, norm_curr, calls
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute Jacobian via forward differences
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_e = F(x_curr + e)
                calls += 1
                J[:, j] = (Fx_e - Fx_curr) / h
            
            # Solve J dx = -Fx
            try:
                dx = -np.linalg.solve(J, Fx_curr)
            except np.linalg.LinAlgError:
                try:
                    dx = -np.linalg.solve(J + 1e-8 * np.eye(2), Fx_curr)
                except np.linalg.LinAlgError:
                    dx = -Fx_curr * 0.01
            
            step_norm = np.linalg.norm(dx)
            if step_norm < 1e-15:
                break
            if step_norm > 1.0:
                dx = dx / step_norm
            
            # Backtracking line search
            alpha = 1.0
            accepted = False
            for _ in range(8):
                x_cand = x_curr + alpha * dx
                F_cand = F(x_cand)
                calls += 1
                norm_cand = np.linalg.norm(F_cand)
                
                if norm_cand <= 1e-8:
                    return x_cand, norm_cand, calls
                
                if norm_cand < best_norm:
                    best_x = x_cand.copy()
                    best_norm = norm_cand
                
                if norm_cand < norm_curr * 0.5:
                    accepted = True
                    x_curr = x_cand
                    Fx_curr = F_cand
                    norm_curr = norm_cand
                    break
                else:
                    alpha *= 0.5
            
            if not accepted:
                # Try small step in negative gradient direction
                x_cand = x_curr - 0.05 * Fx_curr
                F_cand = F(x_cand)
                calls += 1
                norm_cand = np.linalg.norm(F_cand)
                if norm_cand < norm_curr:
                    x_curr = x_cand
                    Fx_curr = F_cand
                    norm_curr = norm_cand
                    if norm_cand < best_norm:
                        best_x = x_cand.copy()
                        best_norm = norm_cand
                else:
                    break
            
            if norm_curr <= 1e-8:
                return x_curr, norm_curr, calls
        
        return x_curr, norm_curr, calls
    
    total_calls = 1  # initial F(x0)
    
    # Phase 1: Newton from x0
    x1, n1, c1 = newton_from(x, Fx, max_iter=25, h=1e-5)
    total_calls += c1 - 1  # c1 includes the initial F call we already counted
    if n1 <= 1e-8:
        return x1
    
    # Phase 2: Multi-start with angular sampling
    # Try multiple starting points around x0 at different radii
    radii = [0.5, 1.0, 2.0, 5.0]
    angles = [i * np.pi / 4 for i in range(8)]  # 8 directions
    
    for r in radii:
        for angle in angles:
            if time.time() > deadline:
                break
            if total_calls > 400:
                break
            
            x_start = x0 + r * np.array([np.cos(angle), np.sin(angle)])
            try:
                F_start = F(x_start)
            except Exception:
                continue
            total_calls += 1
            
            norm_start = np.linalg.norm(F_start)
            if norm_start <= 1e-8:
                return x_start
            if norm_start < best_norm:
                best_x = x_start.copy()
                best_norm = norm_start
            
            # Quick Newton from this start
            x_try, n_try, c_try = newton_from(x_start, F_start, max_iter=15, h=1e-5)
            total_calls += c_try - 1
            
            if n_try <= 1e-8:
                return x_try
    
    # Phase 3: If still not converged, try a few more with different h
    if best_norm > 1e-8:
        for h_try in [1e-4, 1e-6]:
            if time.time() > deadline:
                break
            if total_calls > 450:
                break
            
            x_start = x0
            F_start = F(x_start)
            total_calls += 1
            
            x_try, n_try, c_try = newton_from(x_start, F_start, max_iter=20, h=h_try)
            total_calls += c_try - 1
            
            if n_try <= 1e-8:
                return x_try
    
    return best_x
# EVOLVE-BLOCK-END
