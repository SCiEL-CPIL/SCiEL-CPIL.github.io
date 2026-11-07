"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    x = np.array(x0, dtype=float, copy=True)
    Fx = F(x)
    norm_F = np.linalg.norm(Fx)
    
    if norm_F <= 1e-8:
        return x
    
    best_x = x.copy()
    best_norm = norm_F
    
    # Broyden's method with good initial Jacobian
    # Use central differences for better initial Jacobian estimate
    h = 1e-5
    J = np.zeros((2, 2))
    
    # Compute initial Jacobian using central differences
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fp = F(x + e)
        Fm = F(x - e)
        J[:, j] = (Fp - Fm) / (2 * h)
    
    total_calls = 5  # 1 initial + 4 for central diff Jacobian
    
    max_iter = 50
    polish_threshold = 1e-6
    
    def update_best(x_cand, norm_cand):
        nonlocal best_x, best_norm
        if norm_cand < best_norm:
            best_x = x_cand.copy()
            best_norm = norm_cand
    
    # Main Broyden loop
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        in_polish = norm_F < polish_threshold
        
        # Solve J dx = -Fx
        try:
            dx = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            try:
                dx = -np.linalg.solve(J + 1e-10 * np.eye(2), Fx)
            except np.linalg.LinAlgError:
                # Fallback: steepest descent step
                dx = -Fx / max(norm_F, 1e-10) * 0.1
        
        step_norm = np.linalg.norm(dx)
        if step_norm < 1e-15:
            # Already at a root or J is singular in a bad way
            if norm_F <= 1e-8:
                return x
            # Try a small perturbation
            x_candidate = x + 1e-4 * np.array([1.0, 0.0])
            F_candidate = F(x_candidate)
            norm_candidate = np.linalg.norm(F_candidate)
            total_calls += 1
            if norm_candidate < norm_F:
                x = x_candidate
                Fx = F_candidate
                norm_F = norm_candidate
                update_best(x, norm_F)
            continue
        
        # Limit step size
        if step_norm > 1.0:
            dx = dx / step_norm
        
        if in_polish:
            # Polish mode: try full step, then damped
            accepted = False
            for alpha in [1.0, 0.5, 0.25, 0.1]:
                x_candidate = x + alpha * dx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                total_calls += 1
                
                if norm_candidate <= 1e-8:
                    return x_candidate
                
                update_best(x_candidate, norm_candidate)
                
                if norm_candidate < norm_F:
                    accepted = True
                    s = x_candidate - x
                    y = F_candidate - Fx
                    denom = s @ s
                    if denom > 1e-15:
                        J = J + np.outer(y - J @ s, s) / denom
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    break
            
            if not accepted:
                # Try gradient descent step
                x_candidate = x - 0.01 * Fx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                total_calls += 1
                if norm_candidate < norm_F:
                    s = x_candidate - x
                    y = F_candidate - Fx
                    denom = s @ s
                    if denom > 1e-15:
                        J = J + np.outer(y - J @ s, s) / denom
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    update_best(x, norm_F)
                else:
                    # Recompute Jacobian
                    for j in range(2):
                        e = np.zeros(2)
                        e[j] = h
                        Fp = F(x + e)
                        Fm = F(x - e)
                        J[:, j] = (Fp - Fm) / (2 * h)
                        total_calls += 2
                    break
        else:
            # Damped mode: line search
            alpha = 1.0
            accepted = False
            for _ in range(5):
                x_candidate = x + alpha * dx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                total_calls += 1
                
                if norm_candidate <= 1e-8:
                    return x_candidate
                
                update_best(x_candidate, norm_candidate)
                
                if norm_candidate < norm_F * 0.5:
                    accepted = True
                    s = x_candidate - x
                    y = F_candidate - Fx
                    denom = s @ s
                    if denom > 1e-15:
                        J = J + np.outer(y - J @ s, s) / denom
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    break
                else:
                    alpha *= 0.5
            
            if not accepted:
                # Try smaller gradient step
                x_candidate = x - 0.05 * Fx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                total_calls += 1
                if norm_candidate < norm_F:
                    s = x_candidate - x
                    y = F_candidate - Fx
                    denom = s @ s
                    if denom > 1e-15:
                        J = J + np.outer(y - J @ s, s) / denom
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    update_best(x, norm_F)
                else:
                    # Recompute Jacobian
                    for j in range(2):
                        e = np.zeros(2)
                        e[j] = h
                        Fp = F(x + e)
                        Fm = F(x - e)
                        J[:, j] = (Fp - Fm) / (2 * h)
                        total_calls += 2
                    break
        
        if norm_F <= 1e-8:
            return x
    
    # If main loop didn't converge, try multi-start
    if best_norm > 1e-8:
        # Try several starting points around x0
        radii = [0.1, 0.5, 1.0, 2.0]
        angles = [i * np.pi / 4 for i in range(8)]
        
        for r in radii:
            for angle in angles:
                if time.time() > deadline or total_calls > 450:
                    break
                
                x_start = x0 + r * np.array([np.cos(angle), np.sin(angle)])
                try:
                    F_start = F(x_start)
                    total_calls += 1
                except Exception:
                    continue
                
                norm_start = np.linalg.norm(F_start)
                if norm_start <= 1e-8:
                    return x_start
                update_best(x_start, norm_start)
                
                # Quick Broyden from this start
                x_curr = x_start.copy()
                Fx_curr = F_start.copy()
                norm_curr = norm_start
                
                # Compute Jacobian
                J_local = np.zeros((2, 2))
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h
                    Fp = F(x_curr + e)
                    Fm = F(x_curr - e)
                    J_local[:, j] = (Fp - Fm) / (2 * h)
                    total_calls += 2
                
                for _ in range(10):
                    if time.time() > deadline or total_calls > 480:
                        break
                    
                    try:
                        dx = -np.linalg.solve(J_local, Fx_curr)
                    except np.linalg.LinAlgError:
                        break
                    
                    step_norm = np.linalg.norm(dx)
                    if step_norm < 1e-15 or step_norm > 1.0:
                        dx = dx / max(step_norm, 1e-15)
                        if np.linalg.norm(dx) > 1.0:
                            dx /= np.linalg.norm(dx)
                    
                    alpha = 1.0
                    accepted = False
                    for _ in range(3):
                        x_candidate = x_curr + alpha * dx
                        F_candidate = F(x_candidate)
                        norm_candidate = np.linalg.norm(F_candidate)
                        total_calls += 1
                        
                        if norm_candidate <= 1e-8:
                            return x_candidate
                        
                        update_best(x_candidate, norm_candidate)
                        
                        if norm_candidate < norm_curr:
                            accepted = True
                            s = x_candidate - x_curr
                            y = F_candidate - Fx_curr
                            denom = s @ s
                            if denom > 1e-15:
                                J_local = J_local + np.outer(y - J_local @ s, s) / denom
                            x_curr = x_candidate
                            Fx_curr = F_candidate
                            norm_curr = norm_candidate
                            break
                        else:
                            alpha *= 0.5
                    
                    if not accepted:
                        break
                    
                    if norm_curr <= 1e-8:
                        return x_curr
    
    return best_x
# EVOLVE-BLOCK-END
