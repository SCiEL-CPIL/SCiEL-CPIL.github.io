"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Trust-Region Reflective Nelder-Mead hybrid for anisotropic functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin
        budget = self.budget
        dim = self.dim

        calls = 0
        best_fx = np.inf
        best_x = np.zeros(dim)

        def safe_func(x):
            nonlocal calls, best_fx, best_x
            if calls >= budget:
                return np.inf
            calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x_clipped))
            except Exception:
                fx = np.inf
            if np.isfinite(fx) and fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
            return fx

        def fit_quadratic_model(points, values):
            """Fit quadratic model f(x) ~ c + g'x + 0.5*x'H*x from points."""
            n_pts = len(points)
            if n_pts < dim + 1:
                return None
            X = np.array(points)
            v = np.array(values)
            # Build design matrix for quadratic: [1, x_i, x_i*x_j]
            n = dim
            # Number of quadratic terms: 1 + n + n*(n+1)/2
            n_quad = 1 + n + n * (n + 1) // 2
            A = np.zeros((n_pts, n_quad))
            for i in range(n_pts):
                xi = X[i]
                A[i, 0] = 1.0
                A[i, 1:1+n] = xi
                k = 1 + n
                for a in range(n):
                    for b in range(a, n):
                        A[i, k] = xi[a] * xi[b]
                        k += 1
            # Solve least squares
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, v, rcond=None)
            except np.linalg.LinAlgError:
                return None
            # Extract gradient and Hessian
            g = coeffs[1:1+n].copy()
            H = np.zeros((n, n))
            k = 1 + n
            for a in range(n):
                for b in range(a, n):
                    H[a, b] = coeffs[k]
                    H[b, a] = coeffs[k]
                    k += 1
            c = coeffs[0]
            return c, g, H

        def trust_region_step(x0, c, g, H, delta):
            """Minimize quadratic model within trust region ||s|| <= delta."""
            n = len(x0)
            # Solve (H + lambda*I) s = -g with ||s|| <= delta
            # Use eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(H)
            # Regularize: ensure positive definite
            eigvals_reg = np.maximum(eigvals, 1e-8)
            # Try lambda = 0 first
            def solve_with_lambda(lam):
                A = eigvecs @ np.diag(eigvals_reg + lam) @ eigvecs.T
                try:
                    s = np.linalg.solve(A, -g)
                except np.linalg.LinAlgError:
                    return None
                return s
            
            s = solve_with_lambda(0.0)
            if s is None or np.linalg.norm(s) > delta:
                # Bisection on lambda
                lam_lo, lam_hi = 0.0, 1e10
                for _ in range(50):
                    lam_mid = (lam_lo + lam_hi) / 2
                    s_mid = solve_with_lambda(lam_mid)
                    if s_mid is None:
                        lam_lo = lam_mid
                        continue
                    norm_s = np.linalg.norm(s_mid)
                    if abs(norm_s - delta) < 1e-10 * max(1, delta):
                        s = s_mid
                        break
                    elif norm_s > delta:
                        lam_lo = lam_mid
                    else:
                        lam_hi = lam_mid
                else:
                    s = solve_with_lambda(lam_hi)
                    if s is None:
                        return x0, 0.0
                    # Project onto trust region boundary
                    norm_s = np.linalg.norm(s)
                    if norm_s > delta:
                        s = s * (delta / norm_s)
            # Clip to domain
            x_new = x0 + s
            x_new = np.clip(x_new, -5.0, 5.0)
            # Compute model value at x_new
            s_final = x_new - x0
            model_val = c + g @ s_final + 0.5 * s_final @ H @ s_final
            return x_new, model_val

        # Initialize Nelder-Mead simplex
        simplex = []
        f_values = []
        
        x0 = np.zeros(dim)
        f0 = safe_func(x0)
        simplex.append(x0)
        f_values.append(f0)

        step = 1.0
        for i in range(dim):
            xi = np.zeros(dim)
            xi[i] = step
            fi = safe_func(xi)
            simplex.append(xi)
            f_values.append(fi)

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        n = dim + 1
        iterations = 0
        max_iterations = budget // 2

        # Trust region parameters
        tr_delta = 2.0  # Initial trust region radius
        tr_expand = 2.0
        tr_contract = 0.5

        while calls < budget - n and iterations < max_iterations:
            if time.time() > deadline:
                break
            
            iterations += 1

            # Sort simplex by function value
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]

            # Check convergence
            if f_values[-1] - f_values[0] < 1e-12:
                break

            x_worst = simplex[-1]
            x_best = simplex[0]
            
            # Centroid of all points except the worst
            x_centroid = np.mean(simplex[:-1], axis=0)

            # Try trust-region step from best point using quadratic model
            model = fit_quadratic_model(simplex, f_values)
            used_tr = False
            if model is not None:
                c, g, H = model
                x_tr, model_val = trust_region_step(x_best, c, g, H, tr_delta)
                # Only evaluate if the model predicts improvement
                if model_val < f_values[0] - 1e-10:
                    f_tr = safe_func(x_tr)
                    if np.isfinite(f_tr):
                        # Actual gain
                        actual_gain = f_values[0] - f_tr
                        predicted_gain = f_values[0] - model_val
                        if predicted_gain > 1e-10:
                            rho_tr = actual_gain / predicted_gain
                        else:
                            rho_tr = 0.0
                        
                        if rho_tr > 0.75:
                            # Good agreement: expand trust region and accept
                            simplex[-1] = x_tr
                            f_values[-1] = f_tr
                            tr_delta = min(tr_delta * tr_expand, 10.0)
                            used_tr = True
                        elif rho_tr > 0.25:
                            # Fair agreement: accept but keep trust region
                            simplex[-1] = x_tr
                            f_values[-1] = f_tr
                            used_tr = True
                        else:
                            # Poor agreement: shrink trust region, fall back to NM
                            tr_delta = max(tr_delta * tr_contract, 0.01)

            if not used_tr:
                # Standard Nelder-Mead step
                # 1. Reflection
                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                f_ref = safe_func(x_ref)

                if f_values[0] <= f_ref < f_values[-2]:
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                elif f_ref < f_values[0]:
                    x_exp = x_centroid + gamma * (x_ref - x_centroid)
                    f_exp = safe_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                else:
                    if f_ref < f_values[-1]:
                        x_con = x_centroid + rho * (x_ref - x_centroid)
                    else:
                        x_con = x_centroid - rho * (x_worst - x_centroid)
                    
                    f_con = safe_func(x_con)
                    if f_con < min(f_ref, f_values[-1]):
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        for i in range(1, n):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            f_values[i] = safe_func(simplex[i])
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
