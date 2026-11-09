"""Rewritten optimizer for BBOB f11 (Discus) using a structural change:
derivative-free Gauss-Newton with BFGS Hessian approximation + SA restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free Gauss-Newton (BFGS) with SA restarts for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 270  # safety margin vs 330s kill

        def timed_out():
            return time.time() > deadline

        rng = np.random.default_rng(20240611)

        best_x = None
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            y = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = func(y)
            call_count += 1
            if f < best_f:
                best_f = float(f)
                best_x = y.copy()
            return f

        # ---- Phase 1: diverse initial exploration to locate basin ----
        # Discus has a ring of local minima; we need to find the global one.
        n_init = min(20, budget)
        for k in range(n_init):
            if timed_out() or call_count >= budget:
                break
            if k == 0:
                x0 = np.zeros(dim)
            elif k == 1:
                x0 = 3.5 * rng.standard_normal(dim)
            elif k == 2:
                x0 = np.linspace(-4.0, 4.0, dim)
            elif k == 3:
                x0 = 4.0 * rng.uniform(-1.0, 1.0, dim)
            elif k == 4:
                x0 = 2.5 * rng.standard_normal(dim)
            else:
                x0 = 4.5 * rng.uniform(-1.0, 1.0, dim)
            evaluate(x0)

        if best_x is None:
            return np.zeros(dim), np.inf

        # ---- Phase 2: Derivative-free Gauss-Newton with BFGS ----
        # We maintain a local quadratic model: f(x) ≈ f(xk) + g^T (x-xk) + 0.5 (x-xk)^T H (x-xk)
        # where H is approximated via BFGS updates from finite differences.
        # Steps are computed as Newton steps: s = -H^{-1} g (with damping if needed).

        xk = best_x.copy()
        fk = best_f

        # Initialize Hessian approximation as identity (isotropic)
        H = np.eye(dim)
        # Track previous gradient and step for BFGS
        g_prev = None
        s_prev = None

        # Finite-difference gradient with adaptive step
        def compute_gradient(x, f_center, h_base=1e-4):
            """Compute gradient via central differences. Returns (grad, n_calls)."""
            grad = np.zeros(dim)
            n_calls = 0
            for i in range(dim):
                xi_plus = x.copy()
                xi_plus[i] += h_base
                xi_minus = x.copy()
                xi_minus[i] -= h_base
                fp = evaluate(np.clip(xi_plus, -5.0, 5.0))
                fm = evaluate(np.clip(xi_minus, -5.0, 5.0))
                n_calls += 2
                if timed_out() or call_count >= budget:
                    break
                grad[i] = (fp - fm) / (2.0 * h_base)
            return grad, n_calls

        # Line search along direction
        def line_search(xk, fk, direction, h_base=1e-4):
            """Backtracking line search. Returns (x_new, f_new, step_size, n_calls)."""
            n_calls = 0
            alpha = 1.0
            c1 = 1e-4  # Armijo
            rho = 0.5
            grad_at_xk, n_calls_grad = compute_gradient(xk, fk, h_base)
            n_calls += n_calls_grad
            if timed_out() or call_count >= budget:
                return xk, fk, 0.0, n_calls

            directional_deriv = np.dot(grad_at_xk, direction)
            if directional_deriv >= 0:
                # Direction is not a descent direction; try negative
                direction = -direction
                directional_deriv = -directional_deriv
                if directional_deriv >= 0:
                    return xk, fk, 0.0, n_calls

            for _ in range(30):
                if timed_out() or call_count >= budget:
                    break
                x_trial = np.clip(xk + alpha * direction, -5.0, 5.0)
                f_trial = evaluate(x_trial)
                n_calls += 1
                if f_trial <= fk + c1 * alpha * directional_deriv:
                    return x_trial, f_trial, alpha, n_calls
                alpha *= rho

            return xk, fk, 0.0, n_calls

        # Main Gauss-Newton loop with BFGS
        max_gn_iterations = 200
        no_improve_count = 0
        max_no_improve = 12

        for it in range(max_gn_iterations):
            if timed_out() or call_count >= budget:
                break

            # Compute gradient
            grad, n_calls = compute_gradient(xk, fk)
            if timed_out() or call_count >= budget:
                break

            # Check convergence
            if np.linalg.norm(grad) < 1e-10:
                break

            # Newton step: s = -H^{-1} grad
            try:
                s = np.linalg.solve(H, -grad)
            except np.linalg.LinAlgError:
                s = -grad

            # Damp if step is too large
            step_norm = np.linalg.norm(s)
            if step_norm > 1.0:
                s = s / step_norm * 1.0

            # Line search
            x_new, f_new, alpha_used, n_ls_calls = line_search(xk, fk, s, h_base=1e-4)
            total_calls = n_calls + n_ls_calls

            if alpha_used == 0.0 or f_new >= fk:
                no_improve_count += 1
                # Shrink H to be more isotropic (damping)
                H = 0.5 * H + 0.5 * np.eye(dim) * np.trace(H) / dim
                if no_improve_count >= max_no_improve:
                    break
                # Try a steepest descent step
                s_descent = -grad / (np.linalg.norm(grad) + 1e-12)
                x_new, f_new, alpha_used, n_ls_calls2 = line_search(xk, fk, s_descent)
                if alpha_used > 0 and f_new < fk:
                    no_improve_count = 0
                    s_actual = x_new - xk
                    g_new, _ = compute_gradient(x_new, f_new)
                    y_bfgs = g_new - grad
                    if np.dot(s_actual, y_bfgs) > 1e-10:
                        rho_bfgs = np.dot(s_actual, y_bfgs)
                        H = H - np.outer(s_actual, s_actual) * np.dot(y_bfgs, y_bfgs) / rho_bfgs + np.outer(y_bfgs, y_bfgs)
                        H = 0.5 * (H + H.T)
                    xk, fk = x_new, f_new
                continue
            else:
                no_improve_count = 0

            # BFGS update
            s_actual = x_new - xk
            if np.linalg.norm(s_actual) > 1e-10:
                g_new, _ = compute_gradient(x_new, f_new)
                y_bfgs = g_new - grad
                if np.dot(s_actual, y_bfgs) > 1e-10:
                    rho_bfgs = 1.0 / np.dot(s_actual, y_bfgs)
                    H = H - np.outer(s_actual, s_actual) * np.dot(y_bfgs, y_bfgs) * rho_bfgs + np.outer(y_bfgs, y_bfgs)
                    H = 0.5 * (H + H.T)
                    # Regularize to maintain positive definiteness
                    eigvals = np.linalg.eigvalsh(H)
                    if eigvals.min() < 1e-8:
                        H += (1e-8 - eigvals.min()) * np.eye(dim)

            xk, fk = x_new, f_new

            # Update best
            if fk < best_f:
                best_f = fk
                best_x = xk.copy()

        # ---- Phase 3: Simulated Annealing restarts to escape local minima ----
        if not timed_out() and call_count < budget:
            sa_temperature = max(0.1, (best_f - (best_f - 1.0)) * 0.5)
            # Use a reasonable starting temperature
            sa_temp = 2.0
            cooling_rate = 0.95
            max_sa_steps = min(budget - call_count, 150)
            sa_steps_per_temp = 3

            x_sa = best_x.copy()
            f_sa = best_f

            for temp_idx in range(max_sa_steps // sa_steps_per_temp + 1):
                if timed_out() or call_count >= budget:
                    break
                for _ in range(sa_steps_per_temp):
                    if timed_out() or call_count >= budget:
                        break
                    # Propose a move
                    step_size = sa_temp * 0.5
                    x_proposed = x_sa + step_size * rng.standard_normal(dim)
                    x_proposed = np.clip(x_proposed, -5.0, 5.0)
                    f_proposed = evaluate(x_proposed)

                    # Accept or reject
                    if f_proposed < f_sa:
                        x_sa = x_proposed
                        f_sa = f_proposed
                        if f_sa < best_f:
                            best_f = f_sa
                            best_x = x_sa.copy()
                    else:
                        delta_f = f_proposed - f_sa
                        if delta_f < 50.0 and rng.random() < np.exp(-delta_f / sa_temp):
                            x_sa = x_proposed
                            f_sa = f_proposed

                sa_temp *= cooling_rate

                # If SA found a better point, do a quick GN polish
                if f_sa < best_f - 1e-6 and not timed_out() and call_count < budget:
                    xk = x_sa.copy()
                    fk = f_sa
                    H = np.eye(dim)
                    for _ in range(15):
                        if timed_out() or call_count >= budget:
                            break
                        grad, _ = compute_gradient(xk, fk)
                        if np.linalg.norm(grad) < 1e-10:
                            break
                        try:
                            s = np.linalg.solve(H, -grad)
                        except np.linalg.LinAlgError:
                            s = -grad
                        step_norm = np.linalg.norm(s)
                        if step_norm > 0.5:
                            s = s / step_norm * 0.5
                        x_new, f_new, alpha_used, _ = line_search(xk, fk, s)
                        if alpha_used == 0.0 or f_new >= fk:
                            break
                        s_actual = x_new - xk
                        if np.linalg.norm(s_actual) > 1e-10:
                            g_new, _ = compute_gradient(x_new, f_new)
                            y_bfgs = g_new - grad
                            if np.dot(s_actual, y_bfgs) > 1e-10:
                                rho_bfgs = 1.0 / np.dot(s_actual, y_bfgs)
                                H = H - np.outer(s_actual, s_actual) * np.dot(y_bfgs, y_bfgs) * rho_bfgs + np.outer(y_bfgs, y_bfgs)
                                H = 0.5 * (H + H.T)
                        xk, fk = x_new, f_new
                        if fk < best_f:
                            best_f = fk
                            best_x = xk.copy()

        # ---- Phase 4: Final Nelder-Mead polish if budget remains ----
        if best_x is not None and not timed_out() and call_count < budget - 10:
            from scipy.optimize import minimize

            candidates = [best_x.copy()]
            for _ in range(2):
                candidates.append(best_x + 0.1 * rng.standard_normal(dim))

            for start_x in candidates:
                if call_count >= budget or timed_out():
                    break
                remaining = budget - call_count
                if remaining < 5:
                    break

                def polish_func(x, _start=start_x):
                    nonlocal call_count, best_f, best_x
                    if call_count >= budget or timed_out():
                        return np.inf
                    y = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                    f = func(y)
                    call_count += 1
                    if f < best_f:
                        best_f = float(f)
                        best_x = y.copy()
                    return f

                try:
                    res = minimize(
                        polish_func,
                        np.clip(start_x, -5.0, 5.0).copy(),
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(remaining, 80),
                            'xatol': 1e-9,
                            'fatol': 1e-12,
                            'adaptive': True,
                        }
                    )
                    if res.fun < best_f:
                        best_f = float(res.fun)
                        best_x = np.clip(res.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
