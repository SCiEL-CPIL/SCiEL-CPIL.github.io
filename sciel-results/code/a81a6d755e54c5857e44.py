"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Robust root finder for 2D systems using Broyden's method (rank-1 update)
    to minimize function calls. Falls back to finite-difference Newton if needed.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s hard limit

    def time_ok():
        return time.time() - start_time < DEADLINE

    x = np.array(x0, dtype=float).copy()

    def compute_F(val):
        return np.asarray(F(val), dtype=float)

    best_x = x.copy()
    best_norm = np.inf

    # Initial Jacobian using forward differences (3 calls: F(x), F(x+e1), F(x+e2))
    def forward_diff_jacobian(x_current):
        """Compute Jacobian using forward differences: 3 calls total."""
        Fx = compute_F(x_current)
        h = 1e-7 * max(1.0, np.linalg.norm(x_current))
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (compute_F(x_current + e) - Fx) / h
        return J, Fx

    # Broyden's method with line search
    max_iters = 100
    J, Fx = forward_diff_jacobian(x)
    
    for it in range(max_iters):
        if not time_ok():
            break
        
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x.copy()
        if norm_Fx <= 1e-10:
            break
        
        # Solve J @ dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, perturb it
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break
        
        # Damped step with backtracking line search
        step_size = 1.0
        x_new = x + step_size * dx
        Fx_new = compute_F(x_new)
        norm_Fx_new = np.linalg.norm(Fx_new)
        
        # Check if step is good; if not, halve
        for _ in range(30):
            if norm_Fx_new < norm_Fx:
                break
            step_size *= 0.5
            x_new = x + step_size * dx
            Fx_new = compute_F(x_new)
            norm_Fx_new = np.linalg.norm(Fx_new)
            if step_size < 1e-15:
                break
        
        # Broyden rank-1 update
        s = x_new - x
        y = Fx_new - Fx
        # J_new = J + (y - J@s) @ s.T / (s.T @ s)
        Js = J @ s
        if np.dot(s, s) > 1e-30:
            J = J + (y - Js) @ s / np.dot(s, s)
        
        x = x_new
        Fx = Fx_new
    
    # Check if we converged
    if best_norm <= 1e-8:
        return best_x
    
    # Fallback: Try finite-difference Newton from best_x
    if time_ok():
        x_fd = best_x.copy()
        for it in range(50):
            if not time_ok():
                break
            J_fd, Fx_fd = forward_diff_jacobian(x_fd)
            norm_Fx_fd = np.linalg.norm(Fx_fd)
            if norm_Fx_fd < best_norm:
                best_norm = norm_Fx_fd
                best_x = x_fd.copy()
            if norm_Fx_fd <= 1e-10:
                break
            
            try:
                dx = np.linalg.solve(J_fd, -Fx_fd)
            except np.linalg.LinAlgError:
                break
            
            step_size = 1.0
            x_new = x_fd + step_size * dx
            Fx_new = compute_F(x_new)
            norm_Fx_new = np.linalg.norm(Fx_new)
            for _ in range(20):
                if norm_Fx_new < norm_Fx_fd:
                    break
                step_size *= 0.5
                x_new = x_fd + step_size * dx
                Fx_new = compute_F(x_new)
                norm_Fx_new = np.linalg.norm(Fx_new)
                if step_size < 1e-15:
                    break
            x_fd = x_new
    
    # Final fallback: random restarts with Broyden
    if best_norm > 1e-8 and time_ok():
        for attempt in range(3):
            if not time_ok():
                break
            x_try = best_x + np.random.randn(2) * 0.5
            J_try, Fx_try = forward_diff_jacobian(x_try)
            x_cur = x_try.copy()
            for it in range(30):
                if not time_ok():
                    break
                norm_Fx_cur = np.linalg.norm(Fx_try)
                if norm_Fx_cur < best_norm:
                    best_norm = norm_Fx_cur
                    best_x = x_cur.copy()
                if norm_Fx_cur <= 1e-10:
                    break
                
                try:
                    dx = np.linalg.solve(J_try, -Fx_try)
                except np.linalg.LinAlgError:
                    break
                
                step_size = 1.0
                x_new = x_cur + step_size * dx
                Fx_new = compute_F(x_new)
                norm_Fx_new = np.linalg.norm(Fx_new)
                for _ in range(15):
                    if norm_Fx_new < norm_Fx_cur:
                        break
                    step_size *= 0.5
                    x_new = x_cur + step_size * dx
                    Fx_new = compute_F(x_new)
                    norm_Fx_new = np.linalg.norm(Fx_new)
                    if step_size < 1e-15:
                        break
                
                s = x_new - x_cur
                y = Fx_new - Fx_try
                Js = J_try @ s
                if np.dot(s, s) > 1e-30:
                    J_try = J_try + (y - Js) @ s / np.dot(s, s)
                
                x_cur = x_new
                Fx_try = Fx_new
    
    # Return best finite result
    if np.all(np.isfinite(best_x)):
        return best_x
    return np.array(x0, dtype=float)
# EVOLVE-BLOCK-END
