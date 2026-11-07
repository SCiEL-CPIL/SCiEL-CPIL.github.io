"""Root finder for 3 fixed 2D systems using Broyden's method with homotopy continuation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Primary: Broyden's method (good-first-step) from x0.
    Fallback: Homotopy continuation from a point where F is known to be zero-ish,
              or from a scaled version, then Broyden polish.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    max_calls = 480  # hard cap on F calls to stay under 500

    call_count = [0]

    def F_wrap(x):
        if call_count[0] >= max_calls:
            raise RuntimeError("call budget exceeded")
        call_count[0] += 1
        val = F(x)
        val = np.asarray(val, dtype=float)
        if val.shape != (2,):
            val = val.reshape(2)
        if not np.all(np.isfinite(val)):
            raise ValueError("non-finite F value")
        return val

    def time_ok():
        return time.time() < deadline

    def norm2(v):
        return float(np.dot(v, v))

    def solve_2x2(A, b):
        # Solve A @ x = b for 2x2 A, 2-d b
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-15:
            return None
        x0 = (b[0] * A[1, 1] - A[0, 1] * b[1]) / det
        x1 = (A[0, 0] * b[1] - b[0] * A[1, 0]) / det
        return np.array([x0, x1])

    def broyden_step(x, Fx, J, max_iters=80):
        """
        Broyden's good method. Returns (x, Fx, J, converged, calls_used).
        """
        calls_local = 0
        for it in range(max_iters):
            if not time_ok():
                break
            if norm2(Fx) <= 1e-16:  # ||F||^2 <= 1e-16 => ||F|| <= 1e-8
                return x, Fx, J, True, calls_local
            # Try Newton step using current J
            step = solve_2x2(J, -Fx)
            if step is None:
                # Singular Jacobian; perturb J
                J[0, 0] += 1e-10
                step = solve_2x2(J, -Fx)
                if step is None:
                    break
            # Line search: try full step, then halve
            alpha = 1.0
            for ls in range(10):
                x_new = x + alpha * step
                try:
                    Fx_new = F_wrap(x_new)
                    calls_local += 1
                except (RuntimeError, ValueError):
                    alpha *= 0.5
                    continue
                if norm2(Fx_new) < norm2(Fx):
                    # Update Jacobian (Broyden good)
                    s = x_new - x
                    y = Fx_new - Fx
                    # J_new = J + (y - J@s) s^T / (s^T s)
                    Js = J @ s
                    denom = np.dot(s, s)
                    if denom > 1e-30:
                        J = J + np.outer(y - Js, s) / denom
                    x = x_new
                    Fx = Fx_new
                    break
                else:
                    alpha *= 0.5
            else:
                # Line search failed; try a smaller random perturbation
                break
        return x, Fx, J, norm2(Fx) <= 1e-16, calls_local

    def compute_fd_jacobian(x, Fx, h=1e-7):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            try:
                Fp = F_wrap(x + e)
            except (RuntimeError, ValueError):
                Fp = Fx + h * 1.0  # fallback
            J[:, j] = (Fp - Fx) / h
        return J

    def try_broyden_from(x0_init):
        """
        Try Broyden from x0_init. Returns (x, Fx, J, converged).
        """
        try:
            Fx = F_wrap(x0_init)
        except (RuntimeError, ValueError):
            return None, None, None, False
        J = compute_fd_jacobian(x0_init, Fx)
        x, Fx, J, conv, _ = broyden_step(x0_init, Fx, J)
        return x, Fx, J, conv

    # --- Primary: Broyden from x0 ---
    x0_arr = np.array(x0, dtype=float)
    x, Fx, J, converged = try_broyden_from(x0_arr)
    if converged:
        return x

    # If we got close, return best so far
    if x is not None and Fx is not None and norm2(Fx) <= 1e-16:
        return x

    # --- Fallback 1: Homotopy continuation ---
    # Idea: find a point x_star where F(x_star) is small by searching,
    # then use homotopy H(t, x) = (1-t)*F(x) + t*(x - x_star) = 0
    # Actually simpler: use a damped Newton / continuation from a better start.
    # Alternative: try several random restarts with Broyden.
    
    # Try random restarts in a bounded region around x0
    base = x0_arr.copy()
    best_x = x if x is not None else base
    best_Fx = Fx if Fx is not None else np.array([1e10, 1e10])
    best_norm = norm2(best_Fx) if best_Fx is not None else 1e20

    if best_x is None:
        best_x = base
        best_norm = 1e20

    rng = np.random.default_rng(42)
    for attempt in range(30):
        if not time_ok() or call_count[0] >= max_calls - 10:
            break
        # Scale: try points at various distances from x0
        scale = 10 ** (rng.uniform(-3, 1))
        direction = rng.standard_normal(2)
        direction /= np.linalg.norm(direction)
        x_try = base + scale * direction
        x, Fx, J, conv = try_broyden_from(x_try)
        if x is not None and Fx is not None:
            n = norm2(Fx)
            if n < best_norm:
                best_norm = n
                best_x = x
                best_Fx = Fx
            if conv:
                return x
        if not time_ok():
            break

    # --- Fallback 2: Homotopy continuation from best_x ---
    # Use H(t, x) = (1-t)*F(x) + t*(x - best_x) = 0
    # At t=1: x = best_x, H = 0. We want to go to t=0: F(x)=0.
    # Continuation: start at t=1, x=best_x, decrease t.
    if best_Fx is not None and best_norm < 1e-4:
        t = 1.0
        x_h = best_x.copy()
        # Jacobian of H w.r.t x: (1-t)*J_F + t*I
        # We need J_F at x_h; approximate
        try:
            Fx_h = F_wrap(x_h)
        except (RuntimeError, ValueError):
            Fx_h = best_Fx
        J_F = compute_fd_jacobian(x_h, Fx_h)
        
        for t_step in range(100):
            if not time_ok() or call_count[0] >= max_calls - 5:
                break
            dt = 0.01
            t_new = max(0.0, t - dt)
            # Solve H(t_new, x) = 0 using Newton from x_h
            # H = (1-t_new)*F(x) + t_new*(x - best_x)
            # dH/dx = (1-t_new)*J_F + t_new*I
            I2 = np.eye(2)
            for newton_it in range(20):
                if not time_ok():
                    break
                # Evaluate H and dH
                try:
                    Fx_cur = F_wrap(x_h)
                except (RuntimeError, ValueError):
                    break
                H = (1 - t_new) * Fx_cur + t_new * (x_h - best_x)
                J_H = (1 - t_new) * J_F + t_new * I2
                step = solve_2x2(J_H, -H)
                if step is None:
                    break
                x_new = x_h + step
                try:
                    Fx_new = F_wrap(x_new)
                except (RuntimeError, ValueError):
                    break
                H_new = (1 - t_new) * Fx_new + t_new * (x_new - best_x)
                if norm2(H_new) < norm2(H):
                    x_h = x_new
                    # Update J_F via FD
                    J_F = compute_fd_jacobian(x_h, Fx_new)
                    break
                else:
                    # Halve step
                    step = 0.5 * step
                    x_h = x_h  # keep same, retry
                    break
            t = t_new
            if t == 0.0:
                # Check if F(x_h) is small
                try:
                    Fx_final = F_wrap(x_h)
                except (RuntimeError, ValueError):
                    break
                if norm2(Fx_final) <= 1e-16:
                    return x_h
                if norm2(Fx_final) < best_norm:
                    best_x = x_h
                    best_norm = norm2(Fx_final)
                break

    # Return best found
    if best_x is not None:
        return best_x
    return x0_arr
# EVOLVE-BLOCK-END
