"""Root finder for 3 fixed 2D systems: Broyden with adaptive step and decoupling check."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Check if the system is locally decoupled (diagonal Jacobian).
       If so, solve each 1D component using a secant method (very few calls).
    2. Otherwise, use Broyden's method with rank-1 Jacobian updates.
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

    def solve_1d_secant(f, a, b, max_iters=30):
        """
        1D secant method. f takes a scalar and returns a scalar.
        Returns (x, f_x, converged).
        """
        fa = f(a)
        fb = f(b)
        if abs(fa) <= tol:
            return a, fa, True
        if abs(fb) <= tol:
            return b, fb, True
        
        for _ in range(max_iters):
            if abs(fb - fa) < 1e-15:
                break
            c = b - fb * (b - a) / (fb - fa)
            fc = f(c)
            if abs(fc) <= tol:
                return c, fc, True
            if abs(fc) < abs(fb):
                a, b = b, c
                fa, fb = fb, fc
            else:
                a, b = a, c
                fa, fb = fc, fb
        return b, fb, abs(fb) <= tol

    def try_decoupled(x0_pt, Fx0_pt):
        """
        Check if system is decoupled at x0. If so, solve 1D components.
        Returns (x, Fx, converged) or None if not decoupled.
        """
        h = 1e-6
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        
        Fx_e1 = safe_F(x0_pt + e1)
        Fx_e2 = safe_F(x0_pt + e2)
        
        if Fx_e1 is None or Fx_e2 is None:
            return None
            
        J = np.column_stack([(Fx_e1 - Fx0_pt) / h, (Fx_e2 - Fx0_pt) / h])
        
        # Check if diagonal dominance is strong (off-diagonals are negligible)
        # Use relative threshold: off-diagonal < 1e-4 * max(diagonal magnitudes)
        diag_max = max(abs(J[0, 0]), abs(J[1, 1]), 1e-10)
        if abs(J[0, 1]) < 1e-4 * diag_max and abs(J[1, 0]) < 1e-4 * diag_max:
            # Define 1D functions
            def f1(x1):
                pt = np.array([x1, x0_pt[1]])
                res = safe_F(pt)
                if res is None:
                    return 1e100
                return res[0]
            
            def f2(x2):
                pt = np.array([x0_pt[0], x2])
                res = safe_F(pt)
                if res is None:
                    return 1e100
                return res[1]
            
            # Secant needs two starting points. Use x0 and x0 +/- small step
            x1_sol, f1_val, conv1 = solve_1d_secant(f1, x0_pt[0] - 0.1, x0_pt[0] + 0.1)
            x2_sol, f2_val, conv2 = solve_1d_secant(f2, x0_pt[1] - 0.1, x0_pt[1] + 0.1)
            
            x_sol = np.array([x1_sol, x2_sol])
            Fx_sol = safe_F(x_sol)
            if Fx_sol is not None:
                norm_sq = float(np.dot(Fx_sol, Fx_sol))
                if norm_sq <= tol_sq:
                    return x_sol, Fx_sol, True
            # If not converged to tol, return the best guess? No, let main handle it.
            return None

        return None

    def broyden_solve(x_start, Fx_start, max_iters=30):
        """
        Broyden's method with rank-1 Jacobian updates.
        Returns (x, Fx, converged, calls_used).
        """
        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
        calls_used = 0

        if norm_sq_cur <= tol_sq:
            return x_cur, Fx_cur, True, calls_used

        # Initial Jacobian estimate
        J = compute_jacobian(x_cur, Fx_cur)
        if J is None:
            return x_cur, Fx_cur, False, 2
        calls_used += 2

        for _iter in range(max_iters):
            if remaining() < 1.0:
                break

            try:
                s = np.linalg.solve(J, -Fx_cur)
            except np.linalg.LinAlgError:
                break

            s_norm = np.linalg.norm(s)
            if s_norm < 1e-15:
                break

            # Backtracking line search - more aggressive acceptance
            alpha = 1.0
            accepted = False
            x_try = x_cur
            Fx_try = Fx_cur
            norm_sq_try = norm_sq_cur

            for _ls in range(12):
                x_try = x_cur + alpha * s
                Fx_try = safe_F(x_try)
                calls_used += 1
                if Fx_try is None:
                    alpha *= 0.5
                    continue
                norm_sq_try = float(np.dot(Fx_try, Fx_try))
                if norm_sq_try <= tol_sq:
                    return x_try, Fx_try, True, calls_used
                # Accept if we make any progress (more aggressive)
                if norm_sq_try < norm_sq_cur * 0.95:
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                # Try full step anyway if we're close
                if norm_sq_cur < 1e-4:
                    x_try = x_cur + s
                    Fx_try = safe_F(x_try)
                    calls_used += 1
                    if Fx_try is not None:
                        norm_sq_try = float(np.dot(Fx_try, Fx_try))
                        if norm_sq_try < norm_sq_cur:
                            x_cur = x_try
                            Fx_cur = Fx_try
                            norm_sq_cur = norm_sq_try
                            s_actual = s
                            F_diff = Fx_cur - Fx_start
                            denom = np.dot(s_actual, s_actual)
                            if denom > 1e-30:
                                J = J + np.outer(F_diff - J @ s_actual, s_actual) / denom
                            continue
                break

            # Broyden rank-1 update
            s_actual = x_try - x_cur
            F_diff = Fx_try - Fx_cur
            denom = np.dot(s_actual, s_actual)
            if denom > 1e-30:
                J = J + np.outer(F_diff - J @ s_actual, s_actual) / denom

            x_cur = x_try
            Fx_cur = Fx_try
            norm_sq_cur = norm_sq_try

        return x_cur, Fx_cur, norm_sq_cur <= tol_sq, calls_used

    def polish_newton(x_start, Fx_start, max_iters=15):
        """
        High-precision Newton polish with recomputed Jacobians.
        Uses tighter internal tolerance for super-convergence.
        Returns (x, Fx, converged, calls_used).
        """
        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
        calls_used = 0
        tol_sq_tight = (1e-10) ** 2  # Tighter internal tolerance

        if norm_sq_cur <= tol_sq:
            return x_cur, Fx_cur, True, calls_used

        for _iter in range(max_iters):
            if remaining() < 0.5:
                break

            J = compute_jacobian(x_cur, Fx_cur)
            if J is None:
                break
            calls_used += 2

            try:
                s = np.linalg.solve(J, -Fx_cur)
            except np.linalg.LinAlgError:
                break

            s_norm = np.linalg.norm(s)
            if s_norm < 1e-14:
                break

            x_try = x_cur + s
            Fx_try = safe_F(x_try)
            calls_used += 1
            if Fx_try is None:
                break

            norm_sq_try = float(np.dot(Fx_try, Fx_try))
            if norm_sq_try <= tol_sq:
                return x_try, Fx_try, True, calls_used

            # Accept if we make significant progress toward tight tolerance
            if norm_sq_try < norm_sq_cur * 0.5 or norm_sq_try <= tol_sq_tight:
                x_cur = x_try
                Fx_cur = Fx_try
                norm_sq_cur = norm_sq_try
            elif norm_sq_try < norm_sq_cur:
                # Small progress, accept but be cautious
                x_cur = x_try
                Fx_cur = Fx_try
                norm_sq_cur = norm_sq_try
            else:
                # Try half step
                x_try_half = x_cur + 0.5 * s
                Fx_try_half = safe_F(x_try_half)
                calls_used += 1
                if Fx_try_half is not None:
                    norm_sq_half = float(np.dot(Fx_try_half, Fx_try_half))
                    if norm_sq_half < norm_sq_cur:
                        x_cur = x_try_half
                        Fx_cur = Fx_try_half
                        norm_sq_cur = norm_sq_half
                    else:
                        break
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

    # Try decoupled solution first
    dec_result = try_decoupled(x0_arr, Fx0)
    if dec_result is not None:
        x_dec, Fx_dec, conv_dec = dec_result
        ns_dec = float(np.dot(Fx_dec, Fx_dec))
        if ns_dec < best_norm_sq:
            best_norm_sq = ns_dec
            best_x = x_dec.copy()
        if best_norm_sq <= tol_sq:
            return best_x

    # Primary attempt from x0 with Broyden
    x_sol, Fx_sol, conv, _ = broyden_solve(x0_arr, Fx0, max_iters=25)
    ns = float(np.dot(Fx_sol, Fx_sol)) if Fx_sol is not None else float('inf')
    if ns < best_norm_sq:
        best_norm_sq = ns
        best_x = x_sol.copy()

    # Polish if we got close - use tighter threshold
    if best_norm_sq < 1e-3:
        x_pol, Fx_pol, conv_pol, _ = polish_newton(best_x, Fx_sol, max_iters=15)
        ns_pol = float(np.dot(Fx_pol, Fx_pol)) if Fx_pol is not None else float('inf')
        if ns_pol < best_norm_sq:
            best_norm_sq = ns_pol
            best_x = x_pol.copy()

    if best_norm_sq <= tol_sq:
        return best_x

    # Second polish pass if still not converged but close
    if best_norm_sq < 1e-6 and best_norm_sq > tol_sq:
        x_pol2, Fx_pol2, conv_pol2, _ = polish_newton(best_x, Fx_sol, max_iters=10)
        ns_pol2 = float(np.dot(Fx_pol2, Fx_pol2)) if Fx_pol2 is not None else float('inf')
        if ns_pol2 < best_norm_sq:
            best_norm_sq = ns_pol2
            best_x = x_pol2.copy()
        if best_norm_sq <= tol_sq:
            return best_x

    # Multi-start
    rng = np.random.RandomState(12345)
    max_restarts = 15
    for restart in range(max_restarts):
        if remaining() < 2.0:
            break
        if best_norm_sq <= tol_sq:
            return best_x

        # Generate diverse starting points - more systematic patterns
        if restart == 0:
            x_start = x0_arr + rng.uniform(-2, 2, size=2)
        elif restart == 1:
            x_start = x0_arr * 2.0
        elif restart == 2:
            x_start = x0_arr * 0.5
        elif restart == 3:
            angle = rng.uniform(0, 2 * np.pi)
            x_start = best_x + 1.0 * np.array([np.cos(angle), np.sin(angle)])
        elif restart == 4:
            angle = rng.uniform(0, 2 * np.pi)
            x_start = best_x + 2.0 * np.array([np.cos(angle), np.sin(angle)])
        elif restart == 5:
            x_start = x0_arr + rng.uniform(-5, 5, size=2)
        elif restart == 6:
            angle = rng.uniform(0, 2 * np.pi)
            x_start = best_x + 5.0 * np.array([np.cos(angle), np.sin(angle)])
        elif restart == 7:
            x_start = x0_arr * rng.uniform(0.3, 3.0)
        elif restart == 8:
            x_start = x0_arr + rng.uniform(-10, 10, size=2)
        elif restart == 9:
            angle = rng.uniform(0, 2 * np.pi)
            x_start = best_x + 10.0 * np.array([np.cos(angle), np.sin(angle)])
        elif restart == 10:
            x_start = x0_arr + rng.uniform(-20, 20, size=2)
        elif restart == 11:
            x_start = np.array([x0_arr[0] * 3, x0_arr[1]])
        elif restart == 12:
            x_start = np.array([x0_arr[0], x0_arr[1] * 3])
        elif restart == 13:
            x_start = x0_arr + rng.uniform(-30, 30, size=2)
        else:
            angle = rng.uniform(0, 2 * np.pi)
            x_start = best_x + 15.0 * np.array([np.cos(angle), np.sin(angle)])

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

        # Try decoupled from new start
        dec_result = try_decoupled(x_start, Fx_start)
        if dec_result is not None:
            x_dec, Fx_dec, conv_dec = dec_result
            ns_dec = float(np.dot(Fx_dec, Fx_dec))
            if ns_dec < best_norm_sq:
                best_norm_sq = ns_dec
                best_x = x_dec.copy()
            if best_norm_sq <= tol_sq:
                return best_x

        # Broyden from this start
        x_sol, Fx_sol, conv, _ = broyden_solve(x_start, Fx_start, max_iters=25)
        ns = float(np.dot(Fx_sol, Fx_sol)) if Fx_sol is not None else float('inf')
        if ns < best_norm_sq:
            best_norm_sq = ns
            best_x = x_sol.copy()

        # Polish if close - use tighter threshold
        if best_norm_sq < 1e-3:
            x_pol, Fx_pol, conv_pol, _ = polish_newton(best_x, Fx_sol, max_iters=12)
            ns_pol = float(np.dot(Fx_pol, Fx_pol)) if Fx_pol is not None else float('inf')
            if ns_pol < best_norm_sq:
                best_norm_sq = ns_pol
                best_x = x_pol.copy()

        if best_norm_sq <= tol_sq:
            return best_x

    return best_x
# EVOLVE-BLOCK-END
