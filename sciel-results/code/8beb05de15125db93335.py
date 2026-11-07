"""Root finder for 3 fixed 2D systems: Broyden with high-precision polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Use Broyden's method with rank-1 Jacobian updates (minimizes F-calls).
    2. Backtracking line search on ||F||^2.
    3. Final high-precision polish pass with exact Newton steps.
    4. Multi-start with diverse initial points if needed.
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
        """Compute Jacobian via central differences. Uses 4 F-calls."""
        h = 1e-5
        # Central differences for better accuracy
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        Fx_p1 = safe_F(x + e1)
        Fx_m1 = safe_F(x - e1)
        Fx_p2 = safe_F(x + e2)
        Fx_m2 = safe_F(x - e2)
        if Fx_p1 is None or Fx_m1 is None or Fx_p2 is None or Fx_m2 is None:
            return None
        J = np.column_stack([
            (Fx_p1 - Fx_m1) / (2 * h),
            (Fx_p2 - Fx_m2) / (2 * h)
        ])
        return J

    def broyden_solve(x_start, Fx_start, max_iters=30):
        """
        Broyden's good method with backtracking line search.
        Returns (x, Fx, converged, calls_used).
        """
        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
        calls_used = 0

        if norm_sq_cur <= tol_sq:
            return x_cur, Fx_cur, True, calls_used

        # Initial Jacobian
        J = compute_jacobian(x_cur, Fx_cur)
        if J is None:
            return x_cur, Fx_cur, False, 4
        calls_used += 4

        prev_x = x_cur.copy()
        prev_Fx = Fx_cur.copy()

        for _ in range(max_iters):
            if remaining() < 1.0:
                break

            # Newton step
            try:
                s = np.linalg.solve(J, -Fx_cur)
            except np.linalg.LinAlgError:
                break

            s_norm = np.linalg.norm(s)
            if s_norm < 1e-15:
                break

            # Backtracking line search
            alpha = 1.0
            accepted = False
            x_try = x_cur
            Fx_try = Fx_cur
            norm_sq_try = norm_sq_cur

            for _ls in range(8):
                x_try = x_cur + alpha * s
                Fx_try = safe_F(x_try)
                calls_used += 1
                if Fx_try is None:
                    alpha *= 0.5
                    continue
                norm_sq_try = float(np.dot(Fx_try, Fx_try))
                if norm_sq_try <= tol_sq:
                    return x_try, Fx_try, True, calls_used
                # Armijo condition
                if norm_sq_try <= norm_sq_cur * (1 - 1e-4 * alpha * s_norm):
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                # Try smaller step
                alpha = 0.1
                x_try = x_cur + alpha * s
                Fx_try = safe_F(x_try)
                calls_used += 1
                if Fx_try is None:
                    break
                norm_sq_try = float(np.dot(Fx_try, Fx_try))
                if norm_sq_try < norm_sq_cur:
                    accepted = True
                else:
                    break

            if not accepted:
                break

            # Broyden rank-1 update
            delta_x = x_try - prev_x
            delta_F = Fx_try - prev_Fx
            if np.linalg.norm(delta_x) > 1e-14:
                # J_new = J + (delta_F - J @ delta_x) @ delta_x^T / (delta_x^T @ delta_x)
                correction = (delta_F - J @ delta_x) @ (delta_x / (delta_x @ delta_x))
                J = J + correction

            x_cur = x_try
            Fx_cur = Fx_try
            norm_sq_cur = norm_sq_try
            prev_x = x_cur.copy()
            prev_Fx = Fx_cur.copy()

            if norm_sq_cur <= tol_sq:
                return x_cur, Fx_cur, True, calls_used

            # Recompute Jacobian every 5 iterations or if step is small
            if _ % 5 == 4 or s_norm < 0.01:
                J_new = compute_jacobian(x_cur, Fx_cur)
                if J_new is not None:
                    J = J_new
                    calls_used += 4
                else:
                    break

        return x_cur, Fx_cur, norm_sq_cur <= tol_sq, calls_used

    def polish(x_start, Fx_start, max_iters=10):
        """
        High-precision polish pass using exact Newton steps with fresh Jacobians.
        Returns (x, Fx, converged, calls_used).
        """
        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
        calls_used = 0

        if norm_sq_cur <= tol_sq:
            return x_cur, Fx_cur, True, calls_used

        for _ in range(max_iters):
            if remaining() < 1.0:
                break

            # Fresh Jacobian for high precision
            J = compute_jacobian(x_cur, Fx_cur)
            if J is None:
                break
            calls_used += 4

            try:
                s = np.linalg.solve(J, -Fx_cur)
            except np.linalg.LinAlgError:
                break

            s_norm = np.linalg.norm(s)
            if s_norm < 1e-14:
                break

            # Full step with small backtracking
            alpha = 1.0
            for _ls in range(3):
                x_try = x_cur + alpha * s
                Fx_try = safe_F(x_try)
                calls_used += 1
                if Fx_try is None:
                    alpha *= 0.5
                    continue
                norm_sq_try = float(np.dot(Fx_try, Fx_try))
                if norm_sq_try <= tol_sq:
                    return x_try, Fx_try, True, calls_used
                if norm_sq_try < norm_sq_cur * 0.5:
                    x_cur = x_try
                    Fx_cur = Fx_try
                    norm_sq_cur = norm_sq_try
                    break
                alpha *= 0.5
            else:
                break

            if norm_sq_cur <= tol_sq:
                return x_cur, Fx_cur, True, calls_used

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

    # Primary attempt from x0 using Broyden
    x_sol, Fx_sol, conv, _ = broyden_solve(x0_arr, Fx0, max_iters=25)
    ns = float(np.dot(Fx_sol, Fx_sol)) if Fx_sol is not None else float('inf')
    if ns < best_norm_sq:
        best_norm_sq = ns
        best_x = x_sol.copy()
    
    # Polish pass if close to solution
    if best_norm_sq < 1e-4 and remaining() > 2.0:
        x_pol, Fx_pol, conv_pol, _ = polish(best_x, Fx_sol, max_iters=8)
        ns_pol = float(np.dot(Fx_pol, Fx_pol)) if Fx_pol is not None else float('inf')
        if ns_pol < best_norm_sq:
            best_norm_sq = ns_pol
            best_x = x_pol.copy()
    
    if conv or best_norm_sq <= tol_sq:
        return best_x

    # Multi-start
    rng = np.random.RandomState(12345)
    max_restarts = 8
    for restart in range(max_restarts):
        if remaining() < 3.0:
            break
        if best_norm_sq <= tol_sq:
            return best_x

        # Generate diverse starting points
        if restart % 3 == 0:
            x_start = x0_arr + rng.uniform(-5, 5, size=2)
        elif restart % 3 == 1:
            radius = 0.5 + restart * 0.5
            angle = rng.uniform(0, 2 * np.pi)
            x_start = best_x + radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            x_start = x0_arr * rng.uniform(0.3, 3.0)

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

        # Broyden from new start
        x_sol, Fx_sol, conv, _ = broyden_solve(x_start, Fx_start, max_iters=20)
        ns = float(np.dot(Fx_sol, Fx_sol)) if Fx_sol is not None else float('inf')
        if ns < best_norm_sq:
            best_norm_sq = ns
            best_x = x_sol.copy()
        
        # Polish if close
        if best_norm_sq < 1e-4 and remaining() > 2.0:
            x_pol, Fx_pol, conv_pol, _ = polish(best_x, Fx_sol, max_iters=6)
            ns_pol = float(np.dot(Fx_pol, Fx_pol)) if Fx_pol is not None else float('inf')
            if ns_pol < best_norm_sq:
                best_norm_sq = ns_pol
                best_x = x_pol.copy()
        
        if conv or best_norm_sq <= tol_sq:
            return best_x

    return best_x
# EVOLVE-BLOCK-END
