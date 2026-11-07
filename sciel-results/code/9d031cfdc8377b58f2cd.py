"""Root finder for 3 fixed 2D systems: Broyden with merit line search and periodic refresh."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Evaluate F at x0. If already converged, return immediately.
    2. Use Broyden's good method with:
       - Initial Jacobian via finite differences (2 extra F-calls)
       - Newton step with backtracking line search on ||F||^2
       - Rank-1 Jacobian update
       - Periodic Jacobian recomputation if convergence stalls
    3. If Broyden fails, try multiple restarts with different initial points.
    4. Return best point found within time budget.
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

    def newton_broyden(x_start, Fx_start, max_iters=50):
        """
        Broyden's good method with merit function line search.
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
            return x_cur, Fx_cur, False, 2
        calls_used += 2

        for it in range(max_iters):
            if remaining() < 1.0:
                break

            try:
                s = np.linalg.solve(J, -Fx_cur)
            except np.linalg.LinAlgError:
                # If J is singular, recompute
                J = compute_jacobian(x_cur, Fx_cur)
                if J is None:
                    break
                calls_used += 2
                continue

            s_norm = np.linalg.norm(s)
            if s_norm < 1e-15:
                break

            # Backtracking line search on merit function m(x) = ||F(x)||^2
            alpha = 1.0
            accepted = False
            x_try = x_cur
            Fx_try = Fx_cur
            norm_sq_try = norm_sq_cur

            for _ls in range(10):
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
                if norm_sq_try < norm_sq_cur - 1e-4 * alpha * np.dot(s, Fx_cur):
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                # Try a smaller step or recompute Jacobian
                if it % 5 == 4:
                    J = compute_jacobian(x_cur, Fx_cur)
                    if J is None:
                        break
                    calls_used += 2
                else:
                    break

            # Broyden rank-1 update
            s_vec = alpha * s
            Fx_new = Fx_try
            Fx_old = Fx_cur
            y = Fx_new - Fx_old
            denom = float(np.dot(s_vec, s_vec))
            if denom > 1e-30:
                J = J + (np.outer(y, s_vec) - np.outer(J, s_vec)) / denom

            x_cur = x_try
            Fx_cur = Fx_new
            norm_sq_cur = norm_sq_try

            # Periodic Jacobian refresh every 5 iterations or if step is small
            if it % 5 == 4 or s_norm < 0.01:
                J_new = compute_jacobian(x_cur, Fx_cur)
                if J_new is not None:
                    J = J_new
                    calls_used += 2
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

    # Primary attempt from x0
    x_sol, Fx_sol, conv, _ = newton_broyden(x0_arr, Fx0, max_iters=50)
    ns = float(np.dot(Fx_sol, Fx_sol)) if Fx_sol is not None else float('inf')
    if ns < best_norm_sq:
        best_norm_sq = ns
        best_x = x_sol.copy()
    if conv or best_norm_sq <= tol_sq:
        return best_x

    # Multi-start
    rng = np.random.RandomState(12345)
    max_restarts = 20
    for restart in range(max_restarts):
        if remaining() < 2.0:
            break
        if best_norm_sq <= tol_sq:
            return best_x

        # Generate diverse starting points
        if restart % 4 == 0:
            x_start = x0_arr + rng.uniform(-10, 10, size=2)
        elif restart % 4 == 1:
            radius = 1.0 + restart
            angle = rng.uniform(0, 2 * np.pi)
            x_start = best_x + radius * np.array([np.cos(angle), np.sin(angle)])
        elif restart % 4 == 2:
            x_start = x0_arr * rng.uniform(0.5, 5.0)
        else:
            x_start = rng.uniform(-5, 5, size=2)

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

        x_sol, Fx_sol, conv, _ = newton_broyden(x_start, Fx_start, max_iters=30)
        ns = float(np.dot(Fx_sol, Fx_sol)) if Fx_sol is not None else float('inf')
        if ns < best_norm_sq:
            best_norm_sq = ns
            best_x = x_sol.copy()
        if conv or best_norm_sq <= tol_sq:
            return best_x

    return best_x
# EVOLVE-BLOCK-END
