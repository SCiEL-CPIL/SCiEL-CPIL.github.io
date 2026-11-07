# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using a trust-region dogleg method
    with Broyden-updated Jacobian approximation.
    """
    x = np.array(x0, dtype=float, copy=True)
    n = 2
    max_total_calls = 450  # Slightly lower to ensure we don't hit the limit, we aim for < 100
    call_count = [0]

    def fcounted(z):
        call_count[0] += 1
        out = F(z)
        out = np.asarray(out, dtype=float)
        return out

    # --- Initial evaluation ---
    Fx = fcounted(x)
    if not np.all(np.isfinite(Fx)):
        return x
    fval = 0.5 * np.dot(Fx, Fx)
    if fval <= (1e-8) ** 2:
        return x

    # --- Initial Jacobian via finite differences ---
    def finite_diff_J(z, Fz, h=1e-6):
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            Fp = fcounted(z + e)
            J[:, j] = (Fp - Fz) / h
        return J

    J = finite_diff_J(x, Fx)
    if not np.all(np.isfinite(J)):
        J = finite_diff_J(x, Fx, h=1e-4)
    if not np.all(np.isfinite(J)):
        J = np.eye(n)

    # --- Trust region parameters ---
    delta = 1.0
    delta_min = 1e-12
    delta_max = 1e6
    eta = 0.1

    for iteration in range(100):
        if call_count[0] >= max_total_calls:
            break

        # Check convergence
        if np.linalg.norm(Fx) <= 1e-8:
            return x

        # Gradient and Gauss-Newton Hessian approximation
        g = J.T @ Fx
        H = J.T @ J

        # Solve H p = -g for the Gauss-Newton step
        try:
            p_gn = np.linalg.solve(H, -g)
            if not np.all(np.isfinite(p_gn)):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            try:
                p_gn = np.linalg.lstsq(H, -g, rcond=None)[0]
            except Exception:
                p_gn = -g / (np.linalg.norm(g) + 1e-30)

        # Steepest descent step
        g_norm = np.linalg.norm(g)
        if g_norm > 1e-30:
            p_sd = -g / g_norm
        else:
            p_sd = np.zeros(n)

        # Dogleg step
        gn_norm = np.linalg.norm(p_gn)
        if gn_norm <= delta:
            p = p_gn
        else:
            # Find the point on the line from p_sd to p_gn at distance delta
            # The dogleg path: first go along p_sd to the point closest to p_gn,
            # then go toward p_gn.
            if g_norm > 1e-30:
                t = np.dot(p_gn, p_sd) / g_norm
                # The corner point is t * p_sd (if t > 0)
                corner = t * p_sd if t > 0 else np.zeros(n)
                # Vector from corner to p_gn
                v = p_gn - corner
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-30:
                    # Distance from origin to corner
                    corner_norm = np.linalg.norm(corner)
                    remaining = delta - corner_norm
                    if remaining <= 0:
                        # We're before the corner
                        p = p_sd * (delta / g_norm)
                    else:
                        p = corner + v * (remaining / v_norm)
                else:
                    p = corner
            else:
                p = p_gn * (delta / gn_norm)

        # Predicted reduction
        pred = fval + np.dot(g, p) + 0.5 * np.dot(p, H @ p)

        # Try the step
        x_new = x + p
        Fx_new = fcounted(x_new)
        if not np.all(np.isfinite(Fx_new)):
            delta *= 0.5
            continue
            
        fval_new = 0.5 * np.dot(Fx_new, Fx_new)
        actual = fval - fval_new
        predicted = fval - pred
        
        if predicted <= 0:
            rho = -1.0
        else:
            rho = actual / predicted

        if rho > eta:
            # Accept the step
            s = p
            y = Fx_new - Fx - J @ s
            # Broyden rank-1 update
            s_norm_sq = np.dot(s, s)
            if s_norm_sq > 1e-15:
                J = J + (y - J @ s) @ s.T / s_norm_sq
            
            x = x_new
            Fx = Fx_new
            fval = fval_new
            
            # Grow trust region
            if rho > 0.75:
                delta = min(2.0 * delta, delta_max)
            else:
                delta = min(1.5 * delta, delta_max)
        else:
            # Reject the step, shrink trust region
            delta *= 0.5
            if delta < delta_min:
                # If stuck, try a random perturbation
                x = x + 1e-4 * np.random.randn(n)
                Fx = fcounted(x)
                if not np.all(np.isfinite(Fx)):
                    break
                fval = 0.5 * np.dot(Fx, Fx)
                J = finite_diff_J(x, Fx)
                delta = 1e-2

    # --- Polish phase: Use Newton with current Broyden J if close, else FD ---
    # We have used some calls. Let's try a few Newton steps with the current J.
    # If J is good, this is very fast (1 call per step).
    # If J is bad, we might need to update it.
    
    for _ in range(10):
        if call_count[0] >= max_total_calls:
            break
        
        Fx_norm = np.linalg.norm(Fx)
        if Fx_norm <= 1e-8:
            return x
        
        # Try Newton step with current J
        try:
            dx = np.linalg.solve(J, -Fx)
            if not np.all(np.isfinite(dx)):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            # J is singular or bad, update J via FD
            J = finite_diff_J(x, Fx)
            try:
                dx = np.linalg.solve(J, -Fx)
                if not np.all(np.isfinite(dx)):
                    break
            except np.linalg.LinAlgError:
                break

        x_new = x + dx
        Fx_new = fcounted(x_new)
        if not np.all(np.isfinite(Fx_new)):
            break
            
        if np.linalg.norm(Fx_new) < Fx_norm:
            # Update J via Broyden
            s = dx
            y = Fx_new - Fx - J @ s
            s_norm_sq = np.dot(s, s)
            if s_norm_sq > 1e-15:
                J = J + (y - J @ s) @ s.T / s_norm_sq
            x = x_new
            Fx = Fx_new
        else:
            # Damped Newton
            found = False
            for t in [0.5, 0.25, 0.1]:
                x_trial = x + t * dx
                Fx_trial = fcounted(x_trial)
                if np.all(np.isfinite(Fx_trial)) and np.linalg.norm(Fx_trial) < Fx_norm:
                    x = x_trial
                    Fx = Fx_trial
                    # Update J
                    s = t * dx
                    y = Fx - Fx - J @ s # Wait, Fx is old, Fx_trial is new.
                    # y = F(x_new) - F(x) - J*s
                    y = Fx_trial - Fx - J @ s
                    s_norm_sq = np.dot(s, s)
                    if s_norm_sq > 1e-15:
                        J = J + (y - J @ s) @ s.T / s_norm_sq
                    found = True
                    break
            if not found:
                break

    return x
# EVOLVE-BLOCK-END
