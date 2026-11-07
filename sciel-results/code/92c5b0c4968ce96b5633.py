"""Root finder for 3 fixed 2D systems using Broyden's method with robust restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def time_left():
        return deadline - time.time()

    x = np.array(x0, dtype=float)
    Fx = F(x)
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)
    
    if best_norm <= 1e-10:
        return best_x

    # Compute initial Jacobian using forward differences
    def compute_jacobian(x, Fx):
        J = np.zeros((2, 2))
        h = 1e-6
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - Fx) / h
        return J

    J = compute_jacobian(x, Fx)
    calls = 3 # 1 for Fx, 2 for J

    max_iter = 100
    for _ in range(max_iter):
        if time_left() < 1.0:
            break
            
        # Solve J * s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, recompute Jacobian
            J = compute_jacobian(x, Fx)
            calls += 2
            try:
                s = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break

        # Line search / Backtracking
        lam = 1.0
        x_new = x + s
        F_new = F(x_new)
        calls += 1
        norm_new = np.linalg.norm(F_new)
        
        accepted = False
        # Backtracking if norm doesn't decrease sufficiently
        # We want monotonic decrease in ||F||
        while norm_new >= best_norm and lam > 1e-8:
            lam *= 0.5
            x_new = x + lam * s
            F_new = F(x_new)
            calls += 1
            norm_new = np.linalg.norm(F_new)
            if norm_new < best_norm:
                accepted = True
                break
            if time_left() < 1.0:
                break
        
        if not accepted:
            # If we couldn't find a decrease, try recomputing Jacobian at current x
            # and maybe taking a smaller step or just restarting
            J = compute_jacobian(x, Fx)
            calls += 2
            # Try a smaller step in the same direction? Or just break if stuck?
            # For robustness, let's try one more solve with new J
            try:
                s = np.linalg.solve(J, -Fx)
                x_new = x + 0.5 * s
                F_new = F(x_new)
                calls += 1
                norm_new = np.linalg.norm(F_new)
                if norm_new < best_norm:
                    accepted = True
                else:
                    break
            except np.linalg.LinAlgError:
                break

        if not accepted:
            break

        # Update best
        if norm_new < best_norm:
            best_norm = norm_new
            best_x = x_new.copy()

        if best_norm <= 1e-10:
            return best_x

        # Broyden Update
        # J_new = J + (F_new - Fx - J*s) * s.T / (s.T * s)
        # Note: s used in update is the actual step taken (lam * s_original)
        # Let's be precise. The secant equation is J_new * (x_new - x) = F_new - Fx
        # Broyden update: J_new = J + (F_new - Fx - J * (x_new - x)) * (x_new - x).T / ||x_new - x||^2
        step_taken = x_new - x
        step_norm_sq = np.dot(step_taken, step_taken)
        if step_norm_sq > 1e-15:
            r = F_new - Fx - J @ step_taken
            J = J + np.outer(r, step_taken) / step_norm_sq
            
        x = x_new
        Fx = F_new

    # Final check
    if best_norm <= 1e-8:
        return best_x
    
    # If we didn't get a good enough root, try a few random restarts or just return best
    # Given the strict constraint, returning best_x is the safest fallback
    return best_x
# EVOLVE-BLOCK-END
