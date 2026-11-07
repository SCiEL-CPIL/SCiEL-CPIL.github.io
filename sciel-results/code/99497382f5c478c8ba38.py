import numpy as np
import time

def find_root_system(F, x0):
    tol = 1e-8
    tol_sq = tol * tol
    start_time = time.time()
    time_limit = 148.0

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

    def compute_jacobian_fd(x_pt):
        """Compute Jacobian via central finite differences (4 calls)."""
        h = 1e-5
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

    # Initial Jacobian estimate
    J = compute_jacobian_fd(x)
    if J is None:
        J = np.eye(2)

    # Phase 1: Broyden's method with adaptive step control
    max_iter = 50
    stagnation_count = 0
    max_calls_phase1 = 350

    for _ in range(max_iter):
        if not time_ok() or call_count > max_calls_phase1:
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

        # Adaptive step size - try full step first, then halve
        step_size = 1.0
        x_try = x + step_size * dx
        Fx_try = None
        norm_sq_try = None
        accepted = False

        for _ in range(8):
            if not time_ok() or call_count > max_calls_phase1:
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
            if norm_sq_try < norm_sq * 0.9:
                accepted = True
                break
            step_size *= 0.5
            x_try = x + step_size * dx

        if Fx_try is None or not np.all(np.isfinite(Fx_try)):
            break

        norm_sq_try = float(np.dot(Fx_try, Fx_try))

        # Update Broyden's approximation
        dF = Fx_try - Fx
        actual_dx = step_size * dx
        Jdx = J @ actual_dx
        residual = dF - Jdx
        dx_norm_sq = float(np.dot(actual_dx, actual_dx))
        if dx_norm_sq > 1e-30:
            J = J + np.outer(residual, actual_dx) / dx_norm_sq

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
            if stagnation_count > 4:
                break

    # Phase 2: If not converged, try different strategies
    if best_norm_sq > tol_sq and call_count < 450 and time_ok():
        # Strategy A: Restart Broyden from best_x with fresh Jacobian
        x = best_x.copy()
        try:
            Fx = F_count(x)
        except:
            return best_x
        if not np.all(np.isfinite(Fx)):
            return best_x
        norm_sq = float(np.dot(Fx, Fx))

        if norm_sq < best_norm_sq:
            best_norm_sq = norm_sq
            best_x = x.copy()

        J = compute_jacobian_fd(x)
        if J is None:
            J = np.eye(2)

        # Run Broyden again with more iterations
        stagnation_count = 0
        for _ in range(40):
            if not time_ok() or call_count > 480:
                break
            if norm_sq <= tol_sq:
                return x

            try:
                dx = np.linalg.solve(J, -Fx)
            except:
                J = J + 1e-6 * np.eye(2)
                try:
                    dx = np.linalg.solve(J, -Fx)
                except:
                    break

            step_size = 1.0
            x_try = x + step_size * dx
            Fx_try = None
            norm_sq_try = None

            for _ in range(8):
                if not time_ok() or call_count > 480:
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
                if norm_sq_try < norm_sq * 0.9:
                    break
                step_size *= 0.5
                x_try = x + step_size * dx

            if Fx_try is None or not np.all(np.isfinite(Fx_try)):
                break

            norm_sq_try = float(np.dot(Fx_try, Fx_try))

            dF = Fx_try - Fx
            actual_dx = step_size * dx
            Jdx = J @ actual_dx
            residual = dF - Jdx
            dx_norm_sq = float(np.dot(actual_dx, actual_dx))
            if dx_norm_sq > 1e-30:
                J = J + np.outer(residual, actual_dx) / dx_norm_sq

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
                if stagnation_count > 3:
                    break

    # Phase 3: Polish with high-precision Newton if close to root
    if best_norm_sq > tol_sq and call_count < 490 and time_ok():
        x = best_x.copy()
        try:
            Fx = F_count(x)
        except:
            return best_x
        if not np.all(np.isfinite(Fx)):
            return best_x
        norm_sq = float(np.dot(Fx, Fx))

        if norm_sq < best_norm_sq:
            best_norm_sq = norm_sq
            best_x = x.copy()

        # Use a fresh Jacobian for polishing
        J = compute_jacobian_fd(x)
        if J is None:
            J = np.eye(2)

        for _ in range(20):
            if call_count > 490 or not time_ok():
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
                norm_sq = norm_sq_try
            else:
                break

    return best_x
