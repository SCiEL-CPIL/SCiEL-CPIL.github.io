"""Root finder for 3 fixed 2D systems: structural decoupling and efficient Newton."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Check if the system is locally decoupled (diagonal Jacobian).
       If so, solve each 1D component using a secant method (very few calls).
    2. Otherwise, use a robust 2D Newton method with limited backtracking.
    3. Multi-start with diverse initial points if the first attempt fails.
    """
    x0_arr = np.asarray(x0, dtype=float).copy()
    if x0_arr.shape != (2,) or not np.all(np.isfinite(x0_arr)):
        x0_arr = np.zeros(2)

    tol = 1e-8
    tol_sq = tol * tol
    t0 = time.time()
    deadline = 148.0

    def remaining():
        return deadline - (time.time() - t0)

    def safe_F(x_pt):
        """Call F with finite guard; return None on failure."""
        try:
            y = np.asarray(F(x_pt), dtype=float)
            if y.shape != (2,) or not np.all(np.isfinite(y)):
                return None
            return y
        except Exception:
            return None

    def compute_jacobian(x, Fx):
        """Compute Jacobian via forward differences. Uses 2 F-calls."""
        h = 1e-6
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        Fx_e1 = safe_F(x + e1)
        Fx_e2 = safe_F(x + e2)
        if Fx_e1 is None or Fx_e2 is None:
            return None
        J = np.column_stack([(Fx_e1 - Fx) / h, (Fx_e2 - Fx) / h])
        return J

    def solve_1d_secant(func, x_start, max_iters=20):
        """
        Solves 1D equation func(x) = 0 using secant method.
        func takes a scalar x (index into the 2D vector) and returns the value.
        Actually, to avoid passing complex closures, we will implement this inline 
        or pass the full F and the index.
        """
        pass # Placeholder, implemented in main logic

    def newton_2d(x_start, Fx_start, max_iters=15):
        """
        Robust 2D Newton solver.
        Returns (x, Fx, converged, calls_used).
        """
        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
        calls_used = 0

        if norm_sq_cur <= tol_sq:
            return x_cur, Fx_cur, True, calls_used

        # Check for decoupling
        J = compute_jacobian(x_cur, Fx_cur)
        if J is None:
            return x_cur, Fx_cur, False, 2
        calls_used += 2

        # Check if diagonal dominance is strong
        diag_dominance = (abs(J[0,0]) > 10 * abs(J[0,1]) and abs(J[1,1]) > 10 * abs(J[1,0]))
        
        if diag_dominance:
            # Solve each component separately using secant-like steps
            # We treat F[0](x0, x1) and F[1](x0, x1) as 1D functions
            # Since they are decoupled, we can just do Newton on each 1D part
            # But we need to be careful with the Jacobian entries.
            # Let's just use the diagonal entries as derivatives for 1D Newton
            for _ in range(max_iters):
                if remaining() < 1.0: break
                # Update x0
                if abs(J[0,0]) > 1e-14:
                    dx0 = -Fx_cur[0] / J[0,0]
                    # Limit step
                    if abs(dx0) > 10: dx0 = np.sign(dx0) * 10
                    x_try = x_cur.copy()
                    x_try[0] += dx0
                    Fx_try = safe_F(x_try)
                    calls_used += 1
                    if Fx_try is not None:
                        ns_try = float(np.dot(Fx_try, Fx_try))
                        if ns_try < norm_sq_cur:
                            x_cur = x_try
                            Fx_cur = Fx_try
                            norm_sq_cur = ns_try
                            if norm_sq_cur <= tol_sq:
                                return x_cur, Fx_cur, True, calls_used
                            # Update J diagonal
                            J_new = compute_jacobian(x_cur, Fx_cur)
                            if J_new is not None:
                                J = J_new
                                calls_used += 2
                # Update x1
                if abs(J[1,1]) > 1e-14:
                    dx1 = -Fx_cur[1] / J[1,1]
                    if abs(dx1) > 10: dx1 = np.sign(dx1) * 10
                    x_try = x_cur.copy()
                    x_try[1] += dx1
                    Fx_try = safe_F(x_try)
                    calls_used += 1
                    if Fx_try is not None:
                        ns_try = float(np.dot(Fx_try, Fx_try))
                        if ns_try < norm_sq_cur:
                            x_cur = x_try
                            Fx_cur = Fx_try
                            norm_sq_cur = ns_try
                            if norm_sq_cur <= tol_sq:
                                return x_cur, Fx_cur, True, calls_used
                            # Update J diagonal
                            J_new = compute_jacobian(x_cur, Fx_cur)
                            if J_new is not None:
                                J = J_new
                                calls_used += 2
            return x_cur, Fx_cur, norm_sq_cur <= tol_sq, calls_used

        # Standard 2D Newton
        for _ in range(max_iters):
            if remaining() < 1.0: break
            try:
                s = np.linalg.solve(J, -Fx_cur)
            except np.linalg.LinAlgError:
                break

            s_norm = np.linalg.norm(s)
            if s_norm < 1e-15:
                break

            # Simple backtracking
            alpha = 1.0
            accepted = False
            x_try = x_cur
            Fx_try = Fx_cur
            norm_sq_try = norm_sq_cur
            
            for _ls in range(5):
                x_try = x_cur + alpha * s
                Fx_try = safe_F(x_try)
                calls_used += 1
                if Fx_try is None:
                    alpha *= 0.5
                    continue
                norm_sq_try = float(np.dot(Fx_try, Fx_try))
                if norm_sq_try <= tol_sq:
                    return x_try, Fx_try, True, calls_used
                if norm_sq_try < norm_sq_cur * 0.9:
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                break

            x_cur = x_try
            Fx_cur = Fx_try
            norm_sq_cur = norm_sq_try
            
            # Update Jacobian every 3 steps or if step was small
            if _ % 3 == 2 or s_norm < 0.1:
                J_new = compute_jacobian(x_cur, Fx_cur)
                if J_new is not None:
                    J = J_new
                    calls_used += 2
                else:
                    break

        return x_cur, Fx_cur, norm_sq_cur <= tol_sq, calls_used

    # Main solver
    best_x = x0_arr.copy()
    best_norm_sq = float('inf')

    # Initial evaluation
    Fx0 = safe_F(x0_arr)
    if Fx0 is None:
        return np.zeros(2)
    best_norm_sq = float(np.dot(Fx0, Fx0))
    if best_norm_sq <= tol_sq:
        return x0_arr

    # Primary attempt from x0
    x_sol, Fx_sol, conv, _ = newton_2d(x0_arr, Fx0, max_iters=20)
    ns = float(np.dot(Fx_sol, Fx_sol)) if Fx_sol is not None else float('inf')
    if ns < best_norm_sq:
        best_norm_sq = ns
        best_x = x_sol.copy()
    if conv or best_norm_sq <= tol_sq:
        return best_x

    # Multi-start
    rng = np.random.RandomState(12345)
    max_restarts = 10
    for restart in range(max_restarts):
        if remaining() < 2.0:
            break
        if best_norm_sq <= tol_sq:
            return best_x

        # Generate diverse starting points
        if restart % 3 == 0:
            x_start = x0_arr + rng.uniform(-10, 10, size=2)
        elif restart % 3 == 1:
            radius = 1.0 + restart
            angle = rng.uniform(0, 2 * np.pi)
            x_start = best_x + radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            x_start = x0_arr * rng.uniform(0.5, 5.0)

        if not np.all(np.isfinite(x_start)):
            continue

        Fx_start = safe_F(x_start)
        if Fx_start is None:
            continue

        ns_start = float(np.dot(Fx_start, Fx_start))
        if ns_start < best_norm_sq:
            best_norm_sq = ns_start
            best_x = x_start.copy()
            if best_norm_sq <= tol_sq:
                return best_x

        x_sol, Fx_sol, conv, _ = newton_2d(x_start, Fx_start, max_iters=15)
        ns = float(np.dot(Fx_sol, Fx_sol)) if Fx_sol is not None else float('inf')
        if ns < best_norm_sq:
            best_norm_sq = ns
            best_x = x_sol.copy()
        if conv or best_norm_sq <= tol_sq:
            return best_x

    return best_x
# EVOLVE-BLOCK-END
