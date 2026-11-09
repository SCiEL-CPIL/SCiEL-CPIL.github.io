"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Robust root finder for 2D systems using Broyden's method + Nelder-Mead fallback.
    Strategy:
    1. Broyden's method (quasi-Newton) with line search: 1 F-call/iter after first.
    2. If Broyden fails, Nelder-Mead on ||F(x)||^2 from multiple starting points.
    3. Return best x found with ||F(x)|| <= 1e-8, else best finite x.
    """
    start_time = time.time()
    DEADLINE = 145.0

    def time_ok():
        return time.time() - start_time < DEADLINE

    x = np.array(x0, dtype=float).copy()

    def compute_F(val):
        return np.asarray(F(val), dtype=float)

    best_x = x.copy()
    best_norm = np.inf

    def update_best(x_candidate):
        nonlocal best_x, best_norm
        Fx = compute_F(x_candidate)
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x_candidate.copy()
        return norm_Fx

    # --- Phase 1: Broyden's method (quasi-Newton) ---
    # Initialize Jacobian estimate via finite differences (3 calls)
    h = 1e-7 * max(1.0, np.linalg.norm(x)) ** 0.5
    Fx0 = compute_F(x)
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (compute_F(x + e) - Fx0) / h

    norm_Fx = np.linalg.norm(Fx0)
    update_best(x)

    if norm_Fx <= 1e-8:
        return best_x

    x_broyden = x.copy()
    Fx_broyden = Fx0.copy()
    J_broyden = J.copy()

    for it in range(200):
        if not time_ok():
            break

        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J_broyden, -Fx_broyden)
        except np.linalg.LinAlgError:
            # Fallback: damped least squares
            try:
                dx = np.linalg.lstsq(J_broyden, -Fx_broyden, rcond=None)[0]
            except:
                break

        # Line search
        step = 1.0
        x_new = x_broyden + step * dx
        for _ in range(40):
            Fx_new = compute_F(x_new)
            norm_new = np.linalg.norm(Fx_new)
            if norm_new < np.linalg.norm(Fx_broyden):
                break
            step *= 0.5
            x_new = x_broyden + step * dx
            if step < 1e-15:
                break

        norm_new = np.linalg.norm(Fx_new)
        if norm_new < best_norm:
            best_norm = norm_new
            best_x = x_new.copy()

        if norm_new <= 1e-10:
            break

        # Broyden update (good rank-1)
        s = x_new - x_broyden
        y = Fx_new - Fx_broyden
        if np.linalg.norm(s) > 1e-15:
            J_broyden = J_broyden + np.outer(y - J_broyden @ s, s) / (s @ s)

        x_broyden = x_new
        Fx_broyden = Fx_new

    if best_norm <= 1e-8:
        return best_x

    # --- Phase 2: Nelder-Mead on ||F(x)||^2 from multiple starts ---
    def objective(x_val):
        Fx_val = compute_F(x_val)
        return np.dot(Fx_val, Fx_val)

    # Generate diverse starting points
    start_points = [best_x.copy()]
    base_scale = max(1.0, np.linalg.norm(best_x))
    for i in range(8):
        scale = base_scale * (10 ** ((i % 4) - 1))  # 0.1, 1, 10, 100
        start_points.append(best_x + np.random.randn(2) * scale)

    for sp in start_points:
        if not time_ok():
            break
        if not np.all(np.isfinite(sp)):
            continue

        # Nelder-Mead simplex
        simplex = [sp.copy()]
        for dim in range(2):
            pt = sp.copy()
            pt[dim] += 0.1 * max(1.0, abs(sp[dim]))
            simplex.append(pt)

        # Evaluate simplex
        simplex_f = [objective(p) for p in simplex]

        # Track best within this NM run
        nm_best_x = sp.copy()
        nm_best_f = min(simplex_f)

        for nm_it in range(300):
            if not time_ok():
                break

            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            if simplex_f[0] < nm_best_f:
                nm_best_f = simplex_f[0]
                nm_best_x = simplex[0].copy()

            if np.sqrt(simplex_f[0]) <= 1e-10:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - simplex[-1])
            f_ref = objective(x_ref)

            if f_ref < simplex_f[0]:
                # Expansion
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - simplex[-1])
                f_exp = objective(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    simplex_f[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
            elif f_ref < simplex_f[-2]:
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
            else:
                # Contraction
                rho = 0.5
                x_con = centroid + rho * (simplex[-1] - centroid)
                f_con = objective(x_con)
                if f_con < simplex_f[-1]:
                    simplex[-1] = x_con
                    simplex_f[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, 3):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex_f[i] = objective(simplex[i])

            # Update global best
            if np.sqrt(simplex_f[0]) < best_norm:
                best_norm = np.sqrt(simplex_f[0])
                best_x = simplex[0].copy()

            if best_norm <= 1e-8:
                return best_x

    # Final check
    if np.all(np.isfinite(best_x)):
        return best_x
    return np.array(x0, dtype=float)
# EVOLVE-BLOCK-END
