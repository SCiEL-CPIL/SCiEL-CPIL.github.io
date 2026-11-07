"""Root finder for 3 fixed 2D systems using homotopy continuation with Broyden refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses homotopy continuation from an anchor point, with Broyden refinement.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    call_count = 0
    max_calls = 450  # Leave buffer for safety

    def F_wrapper(x):
        nonlocal call_count
        if call_count >= max_calls:
            return np.array([0.0, 0.0])
        call_count += 1
        try:
            val = F(np.asarray(x, dtype=float))
            val = np.asarray(val, dtype=float).flatten()
            if val.shape != (2):
                val = val.reshape(2)
            if not np.all(np.isfinite(val)):
                return np.array([1e10, 1e10])
            return val
        except Exception:
            return np.array([1e10, 1e10])

    x0 = np.asarray(x0, dtype=float).flatten()
    if x0.shape != (2,):
        x0 = x0.reshape(2)

    best_x = x0.copy()
    best_norm = np.inf

    def _safe_norm(v):
        try:
            n = np.linalg.norm(v)
            if not np.isfinite(n):
                return 1e10
            return n
        except Exception:
            return 1e10

    def _broyden(Fx, x, J, max_iter=20, tol=1e-8):
        """Broyden's method refinement starting from x with Jacobian J."""
        nonlocal call_count
        for _ in range(max_iter):
            if time.time() > deadline or call_count >= max_calls:
                break
            nF = _safe_norm(Fx)
            if nF <= tol:
                return x, Fx, J
            # Solve J dx = -Fx
            try:
                cond = np.linalg.cond(J)
                if cond > 1e12:
                    dx = np.linalg.solve(J + 1e-12 * np.eye(2), -Fx)
                else:
                    dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                dx = -Fx / max(nF, 1e-15)

            # Backtracking line search
            alpha = 1.0
            c1 = 1e-4
            rho = 0.5
            accepted = False
            for _ in range(15):
                if time.time() > deadline or call_count >= max_calls:
                    break
                x_new = x + alpha * dx
                Fx_new = F_wrapper(x_new)
                nF_new = _safe_norm(Fx_new)

                if nF_new < best_norm:
                    nonlocal best_norm, best_x
                    best_norm = nF_new
                    best_x = x_new.copy()

                if nF_new <= nF - c1 * alpha * np.linalg.norm(dx) * nF:
                    # Broyden update
                    if np.linalg.norm(dx) > 1e-15:
                        delta_F = Fx_new - Fx
                        delta_x = x_new - x
                        denom = np.dot(delta_x, delta_x)
                        if denom > 1e-30:
                            J = J + (delta_F[:, None] - J @ delta_x[:, None]) / denom
                    x = x_new
                    Fx = Fx_new
                    accepted = True
                    break
                else:
                    alpha *= rho

            if not accepted:
                break

        return x, Fx, J

    def _compute_jacobian(x, Fx):
        """Compute Jacobian via central differences (4 calls)."""
        nonlocal call_count
        J = np.zeros((2, 2))
        h = 1e-7
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fp = F_wrapper(x + e)
            Fm = F_wrapper(x - e)
            J[:, j] = (Fp - Fm) / (2 * h)
        return J

    # Strategy 1: Try Broyden from x0 first
    Fx0 = F_wrapper(x0)
    norm0 = _safe_norm(Fx0)
    if norm0 < best_norm:
        best_norm = norm0
        best_x = x0.copy()

    if norm0 <= 1e-8:
        return best_x

    if time.time() > deadline or call_count >= max_calls:
        return best_x

    # Compute initial Jacobian
    J0 = _compute_jacobian(x0, Fx0)

    # Run Broyden from x0
    x_b, Fx_b, J_b = _broyden(Fx0, x0, J0, max_iter=25, tol=1e-8)
    nb = _safe_norm(Fx_b)
    if nb < best_norm:
        best_norm = nb
        best_x = x_b.copy()

    if best_norm <= 1e-8:
        return best_x

    if time.time() > deadline or call_count >= max_calls:
        return best_x

    # Strategy 2: Homotopy continuation
    # Anchor: a point where we can solve a simple system
    # We'll use a linear homotopy: H(x, t) = (1-t)*F(x) + t*(x - x_target)
    # where x_target is chosen so that the system is easy to solve at t=1
    # Actually, better: H(x, t) = F(x) + t * (x - x0)
    # At t=0: H = F, at t=1: H = F + (x-x0), which has a solution near x0 if F is not too large

    # Alternative homotopy: blend F with a simple linear function
    # H(x, t) = (1-t)*F(x) + t*(x - a) where a is chosen
    # At t=1, solution is x=a. At t=0, solution is root of F.

    # Let's try: start from a point where we know the solution
    # Use H(x,t) = F(x) + t*(x - x0)
    # At t=0: F(x) = 0
    # At t=large: x ≈ x0 - F(x)/t, so x is close to x0

    # Actually, let's use a different approach:
    # Define G(x, t) = F(x) + t * (x - x_start)
    # At t=0, we want F(x)=0
    # At t=inf, x = x_start
    # We trace from t=large (where x≈x_start) to t=0

    # Start at t = t_max, x = x0 (approximately, since F(x0) + t_max*(x0-x0) = F(x0) ≠ 0)
    # Better: at t=t_max, solve G(x, t_max)=0 => F(x) + t_max*(x-x0) = 0
    # This is a fixed point: x = x0 - F(x)/t_max
    # For large t_max, x ≈ x0 - F(x0)/t_max

    # Let's implement path following:
    t_max = 100.0
    n_steps = 20
    t_vals = np.linspace(t_max, 0.0, n_steps + 1)

    # At t=t_max, approximate solution
    Fx_start = F_wrapper(x0)
    x_path = x0 - Fx_start / t_max

    # Trace the path
    for i in range(len(t_vals) - 1):
        if time.time() > deadline or call_count >= max_calls:
            break

        t_curr = t_vals[i]
        t_next = t_vals[i + 1]
        dt = t_next - t_curr  # negative, decreasing t

        # Define G(x, t) = F(x) + t * (x - x0)
        def G(x, t_val):
            return F_wrapper(x) + t_val * (x - x0)

        # At current point, compute G and its Jacobian
        Gx = G(x_path, t_curr)
        nG = _safe_norm(Gx)

        if nG < best_norm:
            best_norm = nG
            best_x = x_path.copy()

        # Jacobian of G w.r.t. x: J_F + t*I
        # We need J_F, which we can approximate
        J_G = _compute_jacobian(x_path, F_wrapper(x_path)) + t_curr * np.eye(2)

        # Predict step: x_pred = x_path + dt * (dx/dt)
        # dx/dt ≈ -J_G^{-1} * (x - x0)  (from differentiating G(x(t), t)=0)
        # dG/dt = J_F * dx/dt + (x - x0) = 0 => dx/dt = -J_F^{-1}(x-x0)
        # But we have J_G = J_F + t*I, so J_F = J_G - t*I
        J_F_est = J_G - t_curr * np.eye(2)
        try:
            dxdt = np.linalg.solve(-J_F_est, x_path - x0)
        except np.linalg.LinAlgError:
            dxdt = -(x_path - x0) / max(t_curr, 1e-10)

        x_pred = x_path + dt * dxdt

        # Correct: solve G(x, t_next) = 0 near x_pred using Newton
        x_corr = x_pred
        for _ in range(5):
            if time.time() > deadline or call_count >= max_calls:
                break
            Gx_corr = G(x_corr, t_next)
            nG_corr = _safe_norm(Gx_corr)
            if nG_corr < 1e-10:
                break
            J_G_corr = _compute_jacobian(x_corr, F_wrapper(x_corr)) + t_next * np.eye(2)
            try:
                dx = np.linalg.solve(J_G_corr, -Gx_corr)
            except np.linalg.LinAlgError:
                dx = -Gx_corr / max(nG_corr, 1e-15)
            x_corr = x_corr + dx
            if not np.all(np.isfinite(x_corr)):
                break

        x_path = x_corr
        Gx_path = G(x_path, t_next)
        nG_path = _safe_norm(Gx_path)

        if nG_path < best_norm:
            best_norm = nG_path
            best_x = x_path.copy()

        if nG_path <= 1e-8:
            break

    # Final: Broyden refinement from the homotopy endpoint
    if time.time() < deadline and call_count < max_calls:
        Fx_end = F_wrapper(x_path)
        nF_end = _safe_norm(Fx_end)
        if nF_end < best_norm:
            best_norm = nF_end
            best_x = x_path.copy()

        if nF_end <= 1e-8:
            return best_x

        J_end = _compute_jacobian(x_path, Fx_end)
        x_final, Fx_final, _ = _broyden(Fx_end, x_path, J_end, max_iter=20, tol=1e-8)
        nf_final = _safe_norm(Fx_final)
        if nf_final < best_norm:
            best_norm = nf_final
            best_x = x_final.copy()

    return best_x
# EVOLVE-BLOCK-END
