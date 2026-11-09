"""Root-finder for 3 fixed 2D systems using a hybrid Newton + Nelder-Mead approach.
Strategy:
  1. Try Newton's method with finite-difference Jacobian and backtracking line search.
  2. If Newton stagnates or fails, switch to Nelder-Mead on g(x) = 0.5*||F(x)||^2.
  3. When close to the root, polish with Newton steps for high precision.
This hybrid handles non-convex and poorly-conditioned systems better than pure Newton.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 with ||F(x)|| <= 1e-8 (Euclidean norm).
    
    Uses a hybrid approach:
    - Newton's method with FD Jacobian and backtracking for fast local convergence
    - Nelder-Mead on g(x)=0.5*||F(x)||^2 as global fallback
    - Newton polish when close to root
    
    Budget: <500 F-calls per system. Time-limited to ~135s.
    """
    x = np.array(x0, dtype=float, copy=True)
    t_start = time.time()
    deadline = t_start + 135.0
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _time_ok():
        return time.time() < deadline

    def _call_F(x_val):
        nonlocal call_count
        if not _time_ok() or call_count >= 495:
            return None
        call_count += 1
        try:
            result = F(np.asarray(x_val, dtype=float))
        except Exception:
            return None
        result = np.asarray(result, dtype=float)
        if result.shape != (2,):
            try:
                result = result.reshape(2)
            except Exception:
                return None
        if not np.all(np.isfinite(result)):
            return None
        return result

    def _g(x_val):
        """Objective: 0.5 * ||F(x)||^2"""
        Fx = _call_F(x_val)
        if Fx is None:
            return None
        return 0.5 * np.dot(Fx, Fx)

    def _fd_jacobian(x_val, Fx_val, h=1e-5):
        """Compute Jacobian via forward differences. Uses 2 calls."""
        nonlocal call_count
        J = np.zeros((2, 2))
        for j in range(2):
            if not _time_ok() or call_count >= 493:
                break
            e = np.zeros(2)
            e[j] = h
            Fx_plus = _call_F(x_val + e)
            if Fx_plus is not None:
                J[:, j] = (Fx_plus - Fx_val) / h
            else:
                J[:, j] = 1.0
        return J

    def _newton_step(x_val, Fx_val, J_val):
        """One Newton step with backtracking line search. Returns (x_new, Fx_new, converged)."""
        nonlocal call_count
        try:
            if abs(np.linalg.det(J_val)) < 1e-14:
                J_val = J_val + np.eye(2) * 1e-10
            dx = np.linalg.solve(J_val, -Fx_val)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.pinv(J_val) @ (-Fx_val)
            except Exception:
                return x_val, Fx_val, False

        norm_dx = np.linalg.norm(dx)
        if norm_dx < 1e-15:
            return x_val, Fx_val, np.linalg.norm(Fx_val) <= 1e-8

        norm_Fx = np.linalg.norm(Fx_val)
        alpha = 1.0
        for _ in range(20):
            if not _time_ok() or call_count >= 493:
                break
            x_new = x_val + alpha * dx
            if not np.all(np.isfinite(x_new)):
                alpha *= 0.5
                continue
            Fx_new = _call_F(x_new)
            if Fx_new is None:
                break
            norm_new = np.linalg.norm(Fx_new)
            if norm_new <= 1e-8:
                return x_new, Fx_new, True
            if norm_new < norm_Fx * (1 - 1e-4 * alpha):
                return x_new, Fx_new, False
            alpha *= 0.5
            if alpha < 1e-14:
                break
        return x_val, Fx_val, False

    def _nelder_mead(x0_nm, max_iter=200):
        """Nelder-Mead on g(x) = 0.5*||F(x)||^2. Returns (x_best, g_best)."""
        nonlocal call_count
        # Initial simplex
        x0 = np.array(x0_nm, dtype=float)
        g0 = _g(x0)
        if g0 is None:
            return x0, np.inf

        # Build simplex
        simplex = [x0.copy()]
        g_values = [g0]
        for j in range(2):
            e = np.zeros(2)
            e[j] = 0.1 * max(1.0, abs(x0[j]))
            x_new = x0 + e
            g_new = _g(x_new)
            if g_new is None:
                g_new = 1e10
            simplex.append(x_new)
            g_values.append(g_new)

        # Sort
        order = np.argsort(g_values)
        simplex = [simplex[i] for i in order]
        g_values = [g_values[i] for i in order]

        alpha_nm = 1.0   # reflection
        gamma = 2.0      # expansion
        rho = 0.5        # contraction
        sigma = 0.5      # shrink

        for _ in range(max_iter):
            if not _time_ok() or call_count >= 490:
                break

            # Reflect worst point
            x_worst = simplex[-1]
            g_worst = g_values[-1]
            if g_worst <= 1e-16:  # ||F||^2 / 2 <= 1e-16 => ||F|| <= ~1.4e-8
                break

            centroid = np.mean(simplex[:-1], axis=0)
            x_ref = centroid + alpha_nm * (centroid - x_worst)
            g_ref = _g(x_ref)
            if g_ref is None:
                g_ref = 1e10

            if g_ref < g_values[0]:
                # Try expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                g_exp = _g(x_exp)
                if g_exp is None:
                    g_exp = 1e10
                if g_exp < g_ref:
                    simplex[-1] = x_exp
                    g_values[-1] = g_exp
                else:
                    simplex[-1] = x_ref
                    g_values[-1] = g_ref
            elif g_ref < g_values[-2]:
                simplex[-1] = x_ref
                g_values[-1] = g_ref
            else:
                # Contraction
                x_con = centroid + rho * (x_worst - centroid)
                g_con = _g(x_con)
                if g_con is None:
                    g_con = 1e10
                if g_con < g_worst:
                    simplex[-1] = x_con
                    g_values[-1] = g_con
                else:
                    # Shrink
                    for i in range(1, 3):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        g_new = _g(simplex[i])
                        if g_new is None:
                            g_new = 1e10
                        g_values[i] = g_new

            # Re-sort
            order = np.argsort(g_values)
            simplex = [simplex[i] for i in order]
            g_values = [g_values[i] for i in order]

            if g_values[0] <= 1e-16:
                break

        return simplex[0], g_values[0]

    # Initial evaluation
    Fx = _call_F(x)
    if Fx is None:
        return best_x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # Phase 1: Newton's method with FD Jacobian
    J = _fd_jacobian(x, Fx)
    newton_stalled = 0
    prev_norm = norm_Fx

    for it in range(150):
        if not _time_ok() or call_count >= 480:
            break

        x, Fx, converged = _newton_step(x, Fx, J)
        if Fx is None:
            break
        norm_Fx = np.linalg.norm(Fx)

        if converged or norm_Fx <= 1e-8:
            best_x = x.copy()
            best_norm = norm_Fx
            break

        if norm_Fx < best_norm:
            best_x = x.copy()
            best_norm = norm_Fx

        # Update Jacobian via FD if Newton step was accepted
        if norm_Fx < prev_norm * 0.99:
            newton_stalled = 0
            # Update J with FD at new point (2 calls)
            if call_count < 475:
                J = _fd_jacobian(x, Fx)
        else:
            newton_stalled += 1

        prev_norm = norm_Fx

        if newton_stalled >= 10:
            break

    # Phase 2: If not converged, try Nelder-Mead
    if best_norm > 1e-8 and call_count < 470 and _time_ok():
        # Start Nelder-Mead from best point found
        nm_x, nm_g = _nelder_mead(best_x, max_iter=150)
        if nm_g < best_norm ** 2 / 2:
            best_x = nm_x.copy()
            best_norm = np.sqrt(2 * nm_g)

    # Phase 3: Newton polish if close
    if best_norm < 1e-4 and call_count < 490 and _time_ok():
        x = best_x.copy()
        Fx = _call_F(x)
        if Fx is not None:
            norm_Fx = np.linalg.norm(Fx)
            for _ in range(15):
                if not _time_ok() or call_count >= 493:
                    break
                if norm_Fx <= 1e-8:
                    best_x = x.copy()
                    best_norm = norm_Fx
                    break

                h = max(1e-7, 1e-5 * np.sqrt(norm_Fx))
                J = _fd_jacobian(x, Fx, h)
                x, Fx, converged = _newton_step(x, Fx, J)
                if Fx is None:
                    break
                norm_Fx = np.linalg.norm(Fx)
                if norm_Fx < best_norm:
                    best_x = x.copy()
                    best_norm = norm_Fx
                if converged or norm_Fx <= 1e-8:
                    best_x = x.copy()
                    best_norm = norm_Fx
                    break

    # Final check
    if call_count < 497 and _time_ok():
        Fx_final = _call_F(best_x)
        if Fx_final is not None:
            nf = np.linalg.norm(Fx_final)
            if nf <= 1e-8:
                return best_x
            if nf < best_norm:
                best_norm = nf

    return best_x
# EVOLVE-BLOCK-END
