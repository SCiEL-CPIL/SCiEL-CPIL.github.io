# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    x0 = np.asarray(x0, dtype=float).reshape(2)
    start = time.time()
    DEADLINE = 110.0          # safety margin before the 120 s hard kill
    TOL = 1e-8
    MAX_CALLS = 450           # Safety margin below 500
    call_count = 0
    best = x0.copy()
    best_n = float('inf')

    def now_ok():
        return (time.time() - start) < DEADLINE

    def call_guarded(x):
        nonlocal call_count
        if call_count >= MAX_CALLS:
            raise RuntimeError("Call limit exceeded")
        call_count += 1
        res = F(x)
        res = np.asarray(res, dtype=float).reshape(2)
        if not np.all(np.isfinite(res)):
            return res
        return res

    def norm(v):
        return np.linalg.norm(v)

    def update_best(x, n):
        nonlocal best, best_n
        if np.all(np.isfinite(x)) and n < best_n:
            best = x.copy()
            best_n = n

    def broyden(F, x0, max_iter=100):
        x = np.array(x0, dtype=float)
        # Initial Jacobian estimate
        J = np.zeros((2, 2))
        h = 1e-5
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (call_guarded(x + e) - call_guarded(x)) / h
        
        f = call_guarded(x)
        n = norm(f)
        update_best(x, n)
        
        if n <= TOL:
            return x, n

        for _ in range(max_iter):
            if not now_ok() or call_count >= MAX_CALLS:
                break
            
            try:
                # Solve J dx = -f
                dx = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                # If singular, perturb J or break
                break
            
            if not np.all(np.isfinite(dx)):
                break
                
            # Line search / damping
            alpha = 1.0
            found = False
            for _ls in range(20):
                x_new = x + alpha * dx
                if not np.all(np.isfinite(x_new)):
                    alpha *= 0.5
                    continue
                
                f_new = call_guarded(x_new)
                n_new = norm(f_new)
                
                if not np.isfinite(n_new):
                    alpha *= 0.5
                    continue
                    
                # Check for sufficient decrease
                if n_new < n:
                    # Update Broyden
                    # J_new = J + (f_new - f - J*dx) * dx^T / (dx^T * dx)
                    dy = f_new - f
                    J_update = (dy - J @ dx) @ dx / (dx @ dx)
                    J = J + J_update
                    
                    x = x_new
                    f = f_new
                    n = n_new
                    update_best(x, n)
                    found = True
                    break
                else:
                    alpha *= 0.5
            
            if n <= TOL:
                break
            
            if not found:
                # If no step reduces norm, try a small random perturbation or break
                # For robustness, we break and rely on multi-start
                break
                
        return x, n

    # Multi-start strategy
    # Start with x0
    x_res, n_res = broyden(F, x0)
    update_best(x_res, n_res)
    if best_n <= TOL:
        return best

    # Define diverse offsets
    offsets = [
        np.array([0.5, 0.5]), np.array([-0.5, 0.5]),
        np.array([0.5, -0.5]), np.array([-0.5, -0.5]),
        np.array([1.0, 0.0]), np.array([0.0, 1.0]),
        np.array([-1.0, 0.0]), np.array([0.0, -1.0]),
        np.array([2.0, 0.0]), np.array([0.0, 2.0]),
        np.array([-2.0, 0.0]), np.array([0.0, -2.0]),
        np.array([0.1, 0.1]), np.array([-0.1, -0.1]),
    ]
    
    for off in offsets:
        if not now_ok() or call_count >= MAX_CALLS - 10: # Reserve some calls
            break
        x_start = x0 + off
        x_res, n_res = broyden(F, x_start)
        update_best(x_res, n_res)
        if best_n <= TOL:
            return best

    return best
# EVOLVE-BLOCK-END
