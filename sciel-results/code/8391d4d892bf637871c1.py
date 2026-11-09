# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Derivative-Free Trust-Region solver with local quadratic surrogate model."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # State
        x_best = np.zeros(n)
        f_best = np.inf
        calls = 0

        # Trust region
        rho = 1.0
        rho_min = 1e-4
        rho_max = 8.0

        # Model parameters
        c_model = 0.0
        g_model = np.zeros(n)
        H_model = np.zeros((n, n))
        model_ok = False

        # Pool of points for model fitting
        pts = []
        vals = []

        # Helper to evaluate
        def evaluate(x):
            nonlocal x_best, f_best, calls
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            if not np.isfinite(fx):
                fx = 1e6
            calls += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return x, fx

        # Fit quadratic model using points in X, centered at x0
        def do_fit(x0, X_list, F_list):
            nonlocal c_model, g_model, H_model, model_ok
            X = np.array(X_list)
            F = np.array(F_list)
            m = len(X)
            if m < n + 1:
                model_ok = False
                return
            
            # Select points closest to x0 to ensure local validity
            dists = np.linalg.norm(X - x0, axis=1)
            # Filter points within 2*rho of x0 to keep model local
            mask = dists < 2.0 * rho
            Xs = X[mask]
            Fs = F[mask]
            
            if len(Xs) < n + 1:
                # Fallback: use closest n+1 points
                idx = np.argsort(dists)[:n+1]
                Xs = X[idx]
                Fs = F[idx]

            dx = Xs - x0
            # Design matrix: [1, dx, dx^2]
            quad_cols = []
            for i in range(n):
                for j in range(i, n):
                    quad_cols.append(dx[:, i] * dx[:, j])
            
            A = np.hstack([np.ones((len(Xs), 1)), dx, np.column_stack(quad_cols)])
            b = Fs
            
            try:
                coeffs, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
                # Check rank
                if rank < A.shape[1] - 1: # Allow some deficiency
                    model_ok = False
                    return
            except:
                model_ok = False
                return

            c_model = coeffs[0]
            g_model = coeffs[1:1+n].copy()
            
            H = np.zeros((n, n))
            k = 1 + n
            for i in range(n):
                for j in range(i, n):
                    val = coeffs[k]
                    if i == j:
                        H[i, i] += 2.0 * val
                    else:
                        H[i, j] += val
                        H[j, i] += val
                    k += 1
            
            # Regularize Hessian if not PD
            eigvals = np.linalg.eigvalsh(H)
            if np.min(eigvals) < 1e-6:
                lambda_reg = 1e-4 - np.min(eigvals)
                H += lambda_reg * np.eye(n)
            
            H_model = H
            model_ok = True

        # Solve trust region subproblem
        def solve_tr(x0, rho):
            g = g_model
            H = H_model
            
            # Newton step with regularization
            try:
                d_newton = -np.linalg.solve(H, g)
            except:
                # Fallback to steepest descent if solve fails
                norm_g = np.linalg.norm(g)
                if norm_g < 1e-12:
                    return x0
                d_newton = -g / norm_g
                
            norm_d = np.linalg.norm(d_newton)
            if norm_d > rho:
                d_newton = d_newton * (rho / norm_d)
            return x0 + d_newton

        # Main Loop
        # 1. Initialization: Global random sampling to find a good starting point
        # Use first 20 calls for global search
        global_calls = min(20, budget)
        for _ in range(global_calls):
            if calls >= budget:
                break
            x_cur = rng.uniform(-5, 5, n)
            x_cur, f_cur = evaluate(x_cur)
            pts.append(x_cur)
            vals.append(f_cur)
        
        if calls >= budget:
            return x_best, f_best

        # Build initial local set around x_best
        do_fit(x_best, pts, vals)

        # 2. Iterative Optimization
        local_iter = 0
        while calls < budget:
            local_iter += 1
            
            if not model_ok:
                # Fallback: add random points to improve model
                for _ in range(5):
                    if calls >= budget: break
                    dx = rng.standard_normal(n) * rho
                    x_new = x_best + dx
                    x_new, f_new = evaluate(x_new)
                    pts.append(x_new)
                    vals.append(f_new)
                do_fit(x_best, pts, vals)
                if not model_ok:
                    # If still bad, shrink rho and retry
                    rho = max(rho * 0.5, rho_min)
                    continue
            
            # Store old best
            f_old = f_best
            x_old = x_best.copy()
            
            # Solve TR subproblem
            x_cand = solve_tr(x_old, rho)
            x_cand, f_cand = evaluate(x_cand)
            pts.append(x_cand)
            vals.append(f_cand)
            
            # Update Trust Region
            # Predicted reduction
            dx = x_cand - x_old
            pred_red = -(g_model @ dx + 0.5 * dx @ H_model @ dx)
            act_red = f_old - f_cand
            
            if pred_red > 1e-12:
                ratio = act_red / pred_red
            else:
                ratio = 1.0 if act_red > 0 else 0.0

            if ratio > 0.75:
                rho = min(rho * 1.5, rho_max)
            elif ratio > 0.25:
                pass # Keep rho
            else:
                rho = max(rho * 0.5, rho_min)
            
            # Refit model at current best point
            do_fit(x_best, pts, vals)
            
            # Prune points if list gets too large
            if len(pts) > 100:
                X = np.array(pts)
                dists = np.linalg.norm(X - x_best, axis=1)
                mask = dists < 3.0 * rho
                # Also keep the best point always
                best_idx = np.argmin(vals)
                mask[best_idx] = True
                pts = [pts[i] for i in range(len(pts)) if mask[i]]
                vals = [vals[i] for i in range(len(vals)) if mask[i]]

        return x_best, f_best

# EVOLVE-BLOCK-END
