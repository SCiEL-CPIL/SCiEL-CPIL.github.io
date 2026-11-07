import numpy as np
import time


def find_root_system(F, x0):
    """
    Trust-region dogleg root finder for a 2D nonlinear system F: R^2 -> R^2.
    Returns x with ||F(x)|| <= 1e-8 (Euclidean). F is counted per call.
    """
    start_time = time.time()
    deadline = start_time + 120.0

    def elapsed_ok():
        return (time.time() - start_time) < deadline

    x = np.array(x0, dtype=float)
    if x.shape != (2,) or not np.all(np.isfinite(x)):
        x = np.zeros(2)

    Fx = F(x)
    Fx = np.asarray(Fx, dtype=float)
    if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
        Fx = np.zeros(2)
    total_calls = 1

    best_x = x.copy()
    best_norm = float(np.linalg.norm(Fx))

    def update_best(x_c, n_c):
        nonlocal best_x, best_norm
        if np.all(np.isfinite(x_c)) and n_c < best_norm:
            best_x = x_c.copy()
            best_norm = n_c

    def fd_jacobian(xc, Fc):
        # adaptive finite-difference step
        scale = max(1.0, float(np.linalg.norm(xc)))
        h = 1e-5 * scale
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fp = F(xc + e)
            Fp = np.asarray(Fp, dtype=float)
            if Fp.shape != (2,) or not np.all(np.isfinite(Fp)):
                Fp = Fc
            J[:, j] = (Fp - Fc) / h
        return J

    J = fd_jacobian(x, Fx)
    total_calls += 2

    if not np.all(np.isfinite(J)):
        J = np.eye(2)

    def safe_solve(A, b):
        try:
            s = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            s = np.linalg.lstsq(A, b, rcond=None)[0]
        if not np.all(np.isfinite(s)):
            s = np.zeros(2)
        return s

    # ---- main trust-region dogleg loop ----
    max_iter = 200
    trust_radius = 1.0

    for _ in range(max_iter):
        if not elapsed_ok():
            break

        norm_F = float(np.linalg.norm(Fx))
        if norm_F <= 1e-8:
            return x

        # Gauss-Newton direction: J^T J s = -J^T F
        JtF = J.T @ Fx
        H = J.T @ J  # 2x2
        # add tiny ridge for stability
        H = H + 1e-12 * np.eye(2)
        s_gn = -safe_solve(H, JtF)
        g = JtF  # gradient of 0.5*||F||^2

        # steepest descent direction
        gn_norm = float(np.linalg.norm(s_gn))
        if gn_norm < 1e-15:
            # at/near GN optimum; use steepest descent
            g_norm = float(np.linalg.norm(g))
            if g_norm < 1e-15:
                break
            s_sd = -g / g_norm
        else:
            g_norm = float(np.linalg.norm(g))
            if g_norm < 1e-15:
                s_sd = np.zeros(2)
            else:
                # project g onto s_gn to get gamma
                gamma = float(np.dot(g, s_gn)) / (gn_norm * gn_norm)
                s_sd = -g / g_norm
                # dogleg: gamma * s_gn + (s_gn - gamma*s_gn)
                # first leg length
                leg1 = abs(gamma) * gn_norm
                # full GN step length
                leg2 = gn_norm
                # distance from end of leg1 to s_gn
                d = leg2 - leg1
                if d < 0:
                    d = 0.0
                # point on segment at distance trust_radius from start
                if trust_radius <= leg1:
                    s_dog = s_sd * trust_radius
                elif trust_radius >= leg2:
                    s_dog = s_gn
                else:
                    # interpolate along second leg
                    t = (trust_radius - leg1) / max(d, 1e-15)
                    s_dog = s_gn * gamma + (s_gn - s_gn * gamma) * t
                    # equivalently: gamma*s_gn + t*(s_gn - gamma*s_gn)
        if not np.all(np.isfinite(s_dog)):
            s_dog = np.zeros(2)

        # predicted reduction
        pred = -float(np.dot(g, s_dog)) + 0.5 * float(np.dot(s_dog, H @ s_dog))
        if pred <= 1e-15:
            # direction not useful; shrink and try steepest descent
            trust_radius *= 0.5
            s_dog = -g / max(float(np.linalg.norm(g)), 1e-15)
            pred = -float(np.dot(g, s_dog)) + 0.5 * float(np.dot(s_dog, H @ s_dog))
            if pred <= 1e-15:
                break

        x_trial = x + s_dog
        F_trial = F(x_trial)
        F_trial = np.asarray(F_trial, dtype=float)
        total_calls += 1
        if F_trial.shape != (2,) or not np.all(np.isfinite(F_trial)):
            # bad point: shrink trust radius
            trust_radius *= 0.5
            continue

        norm_trial = float(np.linalg.norm(F_trial))
        update_best(x_trial, norm_trial)
        if norm_trial <= 1e-8:
            return x_trial

        # actual reduction in 0.5*||F||^2
        actual = 0.5 * (norm_F * norm_F - norm_trial * norm_trial)
        ratio = actual / pred if pred > 1e-15 else -1.0

        if ratio > 0.75:
            # good step: expand trust radius
            trust_radius = min(10.0, 2.0 * trust_radius)
            accepted = True
        elif ratio > 0.25:
            # acceptable: keep radius
            accepted = True
        else:
            # reject: shrink
            trust_radius = max(1e-8, 0.5 * trust_radius)
            accepted = False

        if accepted:
            s_actual = x_trial - x
            y = F_trial - Fx
            denom = float(s_actual @ s_actual)
            if denom > 1e-15:
                J = J + np.outer(y - J @ s_actual, s_actual) / denom
            x = x_trial
            Fx = F_trial
            norm_F = norm_trial
            if not np.all(np.isfinite(J)):
                J = fd_jacobian(x, Fx)
                total_calls += 2

    # ---- polish phase: Newton steps reusing Broyden J ----
    if elapsed_ok():
        for _ in range(30):
            if not elapsed_ok():
                break
            norm_F = float(np.linalg.norm(Fx))
            if norm_F <= 1e-8:
                return x
            s = -safe_solve(J, Fx)
            if not np.all(np.isfinite(s)):
                break
            sn = float(np.linalg.norm(s))
            if sn < 1e-15:
                break
            # damp if step too large
            if sn > 5.0:
                s = s / sn * 5.0
            x_trial = x + s
            F_trial = F(x_trial)
            F_trial = np.asarray(F_trial, dtype=float)
            total_calls += 1
            if F_trial.shape != (2,) or not np.all(np.isfinite(F_trial)):
                break
            norm_trial = float(np.linalg.norm(F_trial))
            update_best(x_trial, norm_trial)
            if norm_trial <= 1e-8:
                return x_trial
            if norm_trial < norm_F:
                y = F_trial - Fx
                denom = float(s @ s)
                if denom > 1e-15:
                    J = J + np.outer(y - J @ s, s) / denom
                x = x_trial
                Fx = F_trial
                norm_F = norm_trial
            else:
                break

    # final decision: return the best finite point found
    if np.all(np.isfinite(best_x)):
        return best_x
    return x
