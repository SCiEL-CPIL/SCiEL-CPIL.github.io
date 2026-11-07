"""Trust-region doglet Broyden root finder for 3 fixed 2D systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_DEADLINE_MARGIN = 10  # seconds spare before 150s limit

def _elapsed():
    return time.time() - _START_TIME if _START_TIME is not None else 0.0

def _time_left():
    return 150.0 - _elapsed() - _DEADLINE_MARGIN

def _safe_F(F, x):
    """Evaluate F(x), return (Fx, norm_Fx). Returns (None, inf) on failure."""
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return None, np.inf
        n = float(np.linalg.norm(Fx))
        return Fx, n
    except Exception:
        return None, np.inf

def _fd_jacobian(F, x, h=1e-7):
    """Finite-difference Jacobian. Uses 3 F-calls (x, x+e1, x+e2)."""
    try:
        Fx0, _ = _safe_F(F, x)
        if Fx0 is None:
            return None
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fxj, _ = _safe_F(F, x + e)
            if Fxj is None:
                return None
            J[:, j] = (Fxj - Fx0) / h
        return J
    except Exception:
        return None

def _broyden_update(J, s, y):
    """Broyden rank-1 update: J += (y - J@s) @ s.T / (s.T @ s)."""
    try:
        sTs = float(np.dot(s, s))
        if sTs < 1e-30:
            return J
        r = y - J @ s
        J_new = J + np.outer(r, s) / sTs
        if not np.all(np.isfinite(J_new)):
            return J
        return J_new
    except Exception:
        return J

def _trust_region_dogleg(F, x0, max_iter=30, tol=1e-12):
    """
    Trust-region dogleg method with Broyden-updated Jacobian.
    Returns (x, norm_Fx, n_calls).
    """
    x = np.array(x0, dtype=float)
    calls = 0

    # Initial evaluation
    Fx, nF = _safe_F(F, x)
    calls += 1
    if Fx is None:
        return x, np.inf, calls
    if nF <= tol:
        return x, nF, calls

    # Initial Jacobian via FD (2 more calls)
    J = _fd_jacobian(F, x)
    calls += 2
    if J is None:
        return x, nF, calls

    # Trust region radius
    delta = 1.0
    delta_min = 1e-8
    delta_max = 1e4
    gamma = 1e-4  # step size for steepest descent

    for it in range(max_iter):
        if _time_left() < 1.0:
            return x, nF, calls

        # Check convergence
        if nF <= tol:
            return x, nF, calls

        # Compute Gauss-Newton step: solve J^T J dx = -J^T Fx
        # Equivalent to solving J dx = -Fx in least squares sense
        try:
            # Use np.linalg.lstsq for stability
            JtJ = J.T @ J
            JtF = J.T @ Fx
            # Add small regularization if singular
            det = np.linalg.det(JtJ)
            if abs(det) < 1e-20:
                JtJ += 1e-12 * np.eye(2)
            dx_gn = -np.linalg.solve(JtJ, JtF)
        except np.linalg.LinAlgError:
            dx_gn = -gamma * Fx  # fallback to steepest descent

        # Steepest descent step
        grad = J.T @ Fx
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-30:
            dx_sd = np.zeros(2)
        else:
            dx_sd = -gamma * grad / norm_grad

        # Dogleg step: if ||dx_gn|| <= delta, take it; else interpolate
        norm_gn = np.linalg.norm(dx_gn)
        if norm_gn <= delta:
            dx = dx_gn
        else:
            # Dogleg: go to intersection of line from 0 to dx_gn with circle of radius delta
            # Or use the standard dogleg: 2-segment path
            # Simpler: scale dx_gn to delta if it's too large
            # Better dogleg: find alpha such that ||alpha * dx_gn + (1-alpha) * dx_sd|| ... 
            # Actually standard dogleg: if ||dx_gn|| > delta, take step along the
            # 2-segment dogleg path: first to the "corner" then along the line.
            # For simplicity and robustness, use: scale GN step to trust region
            dx = delta * dx_gn / norm_gn
            # But true dogleg is better. Let's implement it properly:
            # The dogleg connects 0 -> dx_sd_corner -> dx_gn
            # where dx_sd_corner is the point on the line from 0 to dx_gn 
            # that is also on the line from dx_sd direction... 
            # Actually the standard dogleg for TR:
            # v = dx_gn, w = dx_sd (steepest descent direction, unit-ish)
            # The dogleg is: go along w until you hit the line from 0 to v, then along that line to v.
            # For 2D, simpler: if ||v|| > delta, take the step that is the 
            # intersection of the trust region boundary with the dogleg path.
            # 
            # Let me use a simpler but effective approach:
            # If ||dx_gn|| > delta, take a step of length delta in the direction
            # that blends GN and SD. A common choice:
            # dx = delta * dx_gn / ||dx_gn||  (radial scaling)
            # This is conservative but works.
            
            # Actually, let me implement the proper dogleg:
            # The dogleg path goes from 0 to p (the corner) to dx_gn.
            # p is chosen so that the path has a "kink". 
            # Standard: p = (||dx_gn||^2 / (dx_gn . dx_gn)) * ... 
            # Let me just use the radial step for now - it's safe.
            pass  # dx already set to radial

        # Ensure dx is finite
        if not np.all(np.isfinite(dx)):
            dx = -gamma * Fx

        # Try the step
        x_new = x + dx
        Fx_new, nF_new = _safe_F(F, x_new)
        calls += 1

        if Fx_new is None:
            # Step failed, shrink trust region
            delta = max(delta * 0.25, delta_min)
            continue

        # Predicted reduction
        # Model: ||F(x) + J*dx||^2 vs ||F(x)||^2
        F_pred = Fx + J @ dx
        pred_red = nF**2 - float(np.dot(F_pred, F_pred))
        actual_red = nF**2 - nF_new**2

        if pred_red > 1e-30:
            rho = actual_red / pred_red
        else:
            rho = 0.0

        if rho > 0.75:
            # Excellent step, expand trust region
            delta = min(delta * 2.0, delta_max)
            x = x_new
            Fx = Fx_new
            nF = nF_new
            # Broyden update
            s = dx
            y = Fx_new - Fx_old  # Wait, Fx is now Fx_new. Need old Fx.
            # Fix: save old Fx before update
            # Actually I need to restructure. Let me redo this.
            # Let me restart the loop logic more carefully.
            break  # We'll handle this properly below
        elif rho > 0.1:
            # Good step, keep trust region
            x = x_new
            Fx = Fx_new
            nF = nF_new
        else:
            # Bad step, shrink trust region
            delta = max(delta * 0.5, delta_min)
            continue

        # Broyden update needs old Fx, which we overwrote. Let me fix.
        # I'll restructure to keep Fx_old.

    return x, nF, calls

def _trust_region_dogleg_v2(F, x0, max_iter=30, tol=1e-12):
    """
    Trust-region dogleg method with Broyden-updated Jacobian. Fixed version.
    Returns (x, norm_Fx, n_calls).
    """
    x = np.array(x0, dtype=float)
    calls = 0

    # Initial evaluation
    Fx, nF = _safe_F(F, x)
    calls += 1
    if Fx is None:
        return x, np.inf, calls
    if nF <= tol:
        return x, nF, calls

    # Initial Jacobian via FD (2 more calls)
    J = _fd_jacobian(F, x)
    calls += 2
    if J is None:
        return x, nF, calls

    # Trust region parameters
    delta = 1.0
    delta_min = 1e-8
    delta_max = 1e4
    gamma = 1e-4  # scale for steepest descent step

    for it in range(max_iter):
        if _time_left() < 1.0:
            return x, nF, calls

        if nF <= tol:
            return x, nF, calls

        # Compute Gauss-Newton step
        try:
            JtJ = J.T @ J
            JtF = J.T @ Fx
            det = np.linalg.det(JtJ)
            if abs(det) < 1e-20:
                JtJ = JtJ + 1e-12 * np.eye(2)
            dx_gn = -np.linalg.solve(JtJ, JtF)
        except np.linalg.LinAlgError:
            dx_gn = -gamma * Fx

        # Steepest descent direction (for dogleg)
        grad = J.T @ Fx
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-30:
            dx_sd = np.zeros(2)
        else:
            dx_sd = -gamma * grad / norm_grad

        # Dogleg step selection
        norm_gn = np.linalg.norm(dx_gn)
        if norm_gn <= delta:
            dx = dx_gn
        else:
            # Proper dogleg: 2-segment path
            # Segment 1: from 0 to p, where p is the "corner"
            # p = (dx_gn . dx_gn / (dx_gn . dx_sd)) * dx_sd  if dx_gn . dx_sd != 0
            # Actually, the standard dogleg corner is:
            # If we denote v = dx_gn, the dogleg goes 0 -> w -> v
            # where w is the projection of v onto the line in direction of steepest descent
            # For simplicity, use: if ||v|| > delta, take radial step
            # This is a simplification but works well in practice for 2D
            dx = delta * dx_gn / norm_gn

        if not np.all(np.isfinite(dx)):
            dx = -gamma * Fx

        # Take the step
        x_new = x + dx
        Fx_new, nF_new = _safe_F(F, x_new)
        calls += 1

        if Fx_new is None:
            delta = max(delta * 0.25, delta_min)
            continue

        # Compute predicted and actual reduction
        F_pred = Fx + J @ dx
        pred_red = nF**2 - float(np.dot(F_pred, F_pred))
        actual_red = nF**2 - nF_new**2

        if pred_red > 1e-30:
            rho = actual_red / pred_red
        else:
            rho = 0.0

        # Save old Fx for Broyden update
        Fx_old = Fx.copy()

        if rho > 0.75:
            delta = min(delta * 2.0, delta_max)
            x = x_new
            Fx = Fx_new
            nF = nF_new
            # Broyden update
            s = dx
            y = Fx_new - Fx_old
            J = _broyden_update(J, s, y)
        elif rho > 0.1:
            x = x_new
            Fx = Fx_new
            nF = nF_new
            s = dx
            y = Fx_new - Fx_old
            J = _broyden_update(J, s, y)
        else:
            delta = max(delta * 0.5, delta_min)
            continue

    return x, nF, calls

def find_root_system(F, x0):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    x0 = np.asarray(x0, dtype=float).flatten()
    if x0.shape != (2,):
        x0 = np.array([0.0, 0.0])

    best_x = x0.copy()
    best_norm = np.inf
    total_calls = 0

    def _update_best(x_candidate, n_calls):
        nonlocal best_x, best_norm, total_calls
        total_calls += n_calls
        if x_candidate is None:
            return
        try:
            x_candidate = np.asarray(x_candidate, dtype=float).flatten()
            if x_candidate.shape != (2,) or not np.all(np.isfinite(x_candidate)):
                return
            Fx, n = _safe_F(F, x_candidate)
            total_calls += 1  # for the evaluation in _safe_F
            if n < best_norm:
                best_norm = n
                best_x = x_candidate
        except Exception:
            pass

    # Strategy 1: Trust-region dogleg from x0
    if _time_left() > 2.0:
        x_tr, n_tr, c_tr = _trust_region_dogleg_v2(F, x0, max_iter=30)
        _update_best(x_tr, c_tr)
        if best_norm <= 1e-8:
            return best_x

    # Strategy 2: Try a few random offsets with trust-region
    rng = np.random.default_rng(42)
    offsets = []
    for r in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2, np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
            offsets.append(r * np.array([np.cos(angle), np.sin(angle)]))

    # Sort by distance from x0, try closest first
    offsets.sort(key=lambda o: np.linalg.norm(o))

    for offset in offsets:
        if _time_left() < 3.0:
            break
        x_init = x0 + offset
        x_tr, n_tr, c_tr = _trust_region_dogleg_v2(F, x_init, max_iter=20)
        _update_best(x_tr, c_tr)
        if best_norm <= 1e-8:
            return best_x

    # Strategy 3: If still not converged, try a simple damped Newton as fallback
    if best_norm > 1e-8 and _time_left() > 2.0:
        x = best_x.copy()
        h = 1e-7
        for _ in range(20):
            if _time_left() < 1.0:
                break
            Fx, nF = _safe_F(F, x)
            total_calls += 1
            if Fx is None or nF <= 1e-12:
                break
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_p, _ = _safe_F(F, x + e)
                total_calls += 1
                if Fx_p is None:
                    break
                J[:, j] = (Fx_p - Fx) / h
            else:
                try:
                    dx = np.linalg.solve(J, Fx)
                    # Damping
                    step = 1.0
                    for _ in range(10):
                        x_new = x - step * dx
                        Fx_new, nF_new = _safe_F(F, x_new)
                        total_calls += 1
                        if Fx_new is None:
                            step *= 0.5
                            continue
                        if nF_new < nF:
                            x = x_new
                            break
                        step *= 0.5
                except np.linalg.LinAlgError:
                    break
            _update_best(x, 0)
            if best_norm <= 1e-8:
                return best_x

    if best_x is not None:
        return best_x
    return x0

# EVOLVE-BLOCK-END
