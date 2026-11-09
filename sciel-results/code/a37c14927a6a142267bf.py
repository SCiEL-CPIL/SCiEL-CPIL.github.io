import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using a robust hybrid strategy:
    1. Try Broyden's method with adaptive step size and line search.
    2. If Newton/Broyden fails, switch to a global search using diverse restarts and a coarse grid.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def time_ok():
        return time.time() < deadline

    x0 = np.array(x0, dtype=float)
    call_count = 0
    MAX_CALLS = 480  # Stay well under 500
    
    def F_counted(x):
        nonlocal call_count
        call_count += 1
        return F(x)
    
    best_x = x0.copy()
    try:
        best_norm = np.linalg.norm(F_counted(x0))
    except:
        best_norm = np.inf
        
    # If we start at a root, we are done
    if best_norm <= 1e-8:
        return x0

    def update_best(x):
        nonlocal best_x, best_norm
        if not np.all(np.isfinite(x)):
            return
        try:
            fx = np.asarray(F_counted(x), dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return
            n = np.linalg.norm(fx)
            if np.isfinite(n) and n < best_norm:
                best_norm = n
                best_x = x.copy()
        except:
            pass

    def broyden_method(x_init, max_iter=30, max_line_search=15):
        """
        Broyden's method with backtracking line search.
        Returns True if a root is found (||F(x)|| <= 1e-8).
        """
        nonlocal call_count
        if not time_ok() or call_count >= MAX_CALLS - 5:
            return False
            
        x = np.array(x_init, dtype=float)
        h = 1e-6
        
        try:
            Fx = np.asarray(F_counted(x), dtype=float)
            if not np.all(np.isfinite(Fx)):
                return False
            n = np.linalg.norm(Fx)
            update_best(x)
            if n <= 1e-8:
                return True
        except:
            return False

        # Compute initial Jacobian
        try:
            Fx1 = np.asarray(F_counted(x + [h, 0]), dtype=float)
            Fx2 = np.asarray(F_counted(x + [0, h]), dtype=float)
            if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
                return False
            J = np.column_stack([(Fx1 - Fx) / h, (Fx2 - Fx) / h])
            
            # Regularize if singular
            try:
                cond = np.linalg.cond(J)
                if cond > 1e12:
                    J = J + np.eye(2) * 1e-8
            except:
                pass
        except:
            return False

        for i in range(max_iter):
            if not time_ok() or call_count >= MAX_CALLS - 2:
                return False
                
            try:
                # Solve J * s = -Fx
                try:
                    s = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                    
                if not np.all(np.isfinite(s)):
                    return False
                    
                # Adaptive step size: try full step, then halve if norm increases
                alpha = 1.0
                success = False
                x_new = x
                Fx_new = Fx
                n_new = n
                
                for _ in range(max_line_search):
                    x_trial = x + alpha * s
                    if not np.all(np.isfinite(x_trial)):
                        alpha *= 0.5
                        continue
                    Fx_trial = np.asarray(F_counted(x_trial), dtype=float)
                    if not np.all(np.isfinite(Fx_trial)):
                        alpha *= 0.5
                        continue
                    n_trial = np.linalg.norm(Fx_trial)
                    if n_trial < n:
                        x_new = x_trial
                        Fx_new = Fx_trial
                        n_new = n_trial
                        success = True
                        update_best(x_new)
                        break
                    alpha *= 0.5
                
                if not success:
                    return False
                    
                if n_new <= 1e-8:
                    return True
                    
                # Broyden Update
                s_actual = x_new - x
                y = Fx_new - Fx
                denom = np.dot(s_actual, s_actual)
                
                if denom > 1e-12:
                    J = J + np.outer(y - J @ s_actual, s_actual) / denom
                else:
                    return False
                    
                x = x_new
                Fx = Fx_new
                n = n_new
            except:
                return False
        return False

    # 1. Try Broyden from x0
    if broyden_method(x0):
        if best_norm <= 1e-8:
            return best_x

    # 2. Generate diverse restarts
    scale = max(1.0, np.linalg.norm(x0))
    restarts = []
    for s in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1), (-1,0), (0,-1), (-1,-1), (-1,1)]:
            restarts.append(x0 + s * np.array([dx, dy]) * scale)
            
    # Add some random-ish perturbations based on x0
    for i in range(20):
        angle = 2 * np.pi * i / 20
        r = scale * (0.5 + i * 0.1)
        restarts.append(x0 + r * np.array([np.cos(angle), np.sin(angle)]))
        
    # Shuffle to avoid bias
    np.random.seed(42)
    np.random.shuffle(restarts)
    
    for x_init in restarts:
        if not time_ok() or call_count >= MAX_CALLS - 5:
            break
        if not np.all(np.isfinite(x_init)):
            continue
        if broyden_method(x_init):
            if best_norm <= 1e-8:
                return best_x

    # 3. Fallback: Grid search if still not found
    # This is expensive, so only do it if we have budget
    if time_ok() and call_count < MAX_CALLS - 50:
        # Define a grid around x0
        center = x0
        radius = max(10.0, 5 * np.linalg.norm(x0))
        steps = 10
        dx = 2 * radius / steps
        
        for i in range(steps + 1):
            for j in range(steps + 1):
                if not time_ok() or call_count >= MAX_CALLS - 5:
                    break
                x_g = center + np.array([i * dx - radius, j * dx - radius])
                if broyden_method(x_g):
                    if best_norm <= 1e-8:
                        return best_x

    return best_x if np.all(np.isfinite(best_x)) else x0
