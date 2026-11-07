import numpy as np
import time

def find_root_system(F, x0):
    tol = 1e-8
    tol_sq = tol * tol
    start_time = time.time()
    time_limit = 145.0

    def time_ok():
        return time.time() - start_time < time_limit

    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)

    call_count = 0

    def F_count(x_pt):
        nonlocal call_count
        call_count += 1
        return F(x_pt)

    # Initial evaluation
    try:
        Fx = F_count(x)
    except:
        return np.zeros(2)

    if not np.all(np.isfinite(Fx)):
        return x

    norm_sq = float(np.dot(Fx, Fx))
    if norm_sq <= tol_sq:
        return x

    best_x = x.copy()
    best_norm_sq = norm_sq

    def compute_jacobian_fd(x_pt, Fx_pt):
        """Compute Jacobian via central finite differences (4 calls)."""
        h = 1e-6
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        try:
            Fx1 = F_count(x_pt + e1)
            Fx2 = F_count(x_pt - e1)
            Fx3 = F_count(x_pt + e2)
            Fx4 = F_count(x_pt - e2)
            J = np.array([(Fx1 - Fx2) / (2*h), (Fx3 - Fx4) / (2*h)]).T
            if not np.all(np.isfinite(J)):
                return None
            return J
        except:
            return None

    # Initial Jacobian estimate using central differences
    J = compute_jacobian_fd(x, Fx)
    if J is None:
        J = np.eye(2)

    # Phase 1: Broyden's method with adaptive step control
    max_iter = 30
    stagnation_count = 0

    for _ in range(max_iter):
        if not time_ok() or call_count > 400:
            break

        if norm_sq <= tol_sq:
            return x

        # Compute Newton step
        try:
            dx = np.linalg.solve(J, -Fx)
        except:
            J = J + 1e-6 * np.eye(2)
            try:
                dx = np.linalg.solve(J, -Fx)
            except:
                break

        # Adaptive step size
        step_size = 1.0
        x_try = x + step_size * dx
        Fx_try = None
        norm_sq_try = None

        for _ in range(10):
            if not time_ok() or call_count > 400:
                break
            try:
                Fx_try = F_count(x_try)
            except:
                break
            if not np.all(np.isfinite(Fx_try)):
                step_size *= 0.5
                x_try = x + step_size * dx
                continue
            norm_sq_try = float(np.dot(Fx_try, Fx_try))
            if norm_sq_try < norm_sq * 0.5:
                break
            step_size *= 0.5
            x_try = x + step_size * dx

        if Fx_try is None or not np.all(np.isfinite(Fx_try)):
            break

        norm_sq_try = float(np.dot(Fx_try, Fx_try))

        # Update Broyden's approximation
        dF = Fx_try - Fx
        Jdx = J @ (step_size * dx)
        residual = dF - Jdx
        dx_norm_sq = float(np.dot(step_size * dx, step_size * dx))
        if dx_norm_sq > 1e-30:
            J = J + np.outer(residual, step_size * dx) / dx_norm_sq

        # Accept if improving
        if norm_sq_try < norm_sq:
            x = x_try
            Fx = Fx_try
            norm_sq = norm_sq_try
            stagnation_count = 0
            if norm_sq < best_norm_sq:
                best_norm_sq = norm_sq
                best_x = x.copy()
        else:
            stagnation_count += 1
            if stagnation_count > 5:
                break

    # Phase 2: If not converged, try homotopy continuation from best_x
    if best_norm_sq > tol_sq and call_count < 450 and time_ok():
        x = best_x.copy()
        try:
            Fx = F_count(x)
        except:
            return best_x
        if not np.all(np.isfinite(Fx)):
            return best_x
        norm_sq = float(np.dot(Fx, Fx))

        # Recompute Jacobian
        J = compute_jacobian_fd(x, Fx)
        if J is None:
            J = np.eye(2)

        # Homotopy: H(x, t) = t*F(x) + (1-t)*(x - x0)
        t = 0.0
        dt = 0.2
        max_homotopy_steps = 20

        for step in range(max_homotopy_steps):
            if not time_ok() or call_count > 450:
                break

            t_target = min(1.0, t + dt)

            # Newton on H
            converged = False
            for _ in range(8):
                if call_count > 450:
                    break

                try:
                    Fx_curr = F_count(x)
                except:
                    break
                if not np.all(np.isfinite(Fx_curr)):
                    break

                H = t_target * Fx_curr + (1.0 - t_target) * (x - x0)
                norm_H = np.linalg.norm(H)
                if norm_H < 1e-12:
                    converged = True
                    break

                J_H = t_target * J + (1.0 - t_target) * np.eye(2)
                try:
                    dx = np.linalg.solve(J_H, -H)
                except:
                    J_H = J_H + 1e-6 * np.eye(2)
                    try:
                        dx = np.linalg.solve(J_H, -H)
                    except:
                        break

                x_try = x + dx
                try:
                    Fx_try = F_count(x_try)
                except:
                    break
                if not np.all(np.isfinite(Fx_try)):
                    break

                # Update J
                dF = Fx_try - Fx_curr
                Jdx = J @ dx
                residual = dF - Jdx
                dx_norm_sq = float(np.dot(dx, dx))
                if dx_norm_sq > 1e-30:
                    J = J + np.outer(residual, dx) / dx_norm_sq

                H_new = t_target * Fx_try + (1.0 - t_target) * (x_try - x0)
                norm_H_new = np.linalg.norm(H_new)

                if norm_H_new < norm_H * 0.5 or norm_H_new < 1e-12:
                    x = x_try
                    Fx = Fx_try
                    if norm_H_new < 1e-12:
                        converged = True
                        break

            if converged:
                t = t_target
                dt = min(0.5, dt * 1.5)
                if t >= 0.999:
                    break
            else:
                dt *= 0.5
                if dt < 1e-3:
                    break

        # Phase 3: Polish with high-precision Newton if close to root
        if t >= 0.999 and call_count < 480 and time_ok():
            for _ in range(15):
                if call_count > 480 or not time_ok():
                    break
                try:
                    Fx_curr = F_count(x)
                except:
                    break
                if not np.all(np.isfinite(Fx_curr)):
                    break
                norm_sq_curr = float(np.dot(Fx_curr, Fx_curr))
                if norm_sq_curr <= tol_sq:
                    return x
                if norm_sq_curr < best_norm_sq:
                    best_norm_sq = norm_sq_curr
                    best_x = x.copy()

                try:
                    dx = np.linalg.solve(J, -Fx_curr)
                except:
                    J = J + 1e-6 * np.eye(2)
                    try:
                        dx = np.linalg.solve(J, -Fx_curr)
                    except:
                        break

                x_try = x + dx
                try:
                    Fx_try = F_count(x_try)
                except:
                    break
                if not np.all(np.isfinite(Fx_try)):
                    break
                norm_sq_try = float(np.dot(Fx_try, Fx_try))

                dF = Fx_try - Fx_curr
                Jdx = J @ dx
                residual = dF - Jdx
                dx_norm_sq = float(np.dot(dx, dx))
                if dx_norm_sq > 1e-30:
                    J = J + np.outer(residual, dx) / dx_norm_sq

                if norm_sq_try < norm_sq_curr:
                    x = x_try
                    Fx = Fx_try
                else:
                    break

    return best_x
