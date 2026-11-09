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

        # --- State ---
        # Pool of evaluated points: X (m, n), F (m,)
        X = np.empty((0, n))
        F = np.empty(0)

        # Current best
        x_best = np.zeros(n)
        f_best = np.inf

        # Trust region radius
        rho = 1.0
        rho_min = 1e-4
        rho_max = 8.0

        # Model parameters (fitted)
        model_ok = False
        c_model = 0.0
        g_model = np.zeros(n)
        H_model = np.zeros((n, n))

        # Step size for fallback
        step_scale = 1.0

        # --- Helper: evaluate and record ---
        def evaluate(x):
            nonlocal x_best, f_best
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            if not np.isfinite(fx):
                fx = 1e6
            # Record in pool
            X = np.vstack([X, x.reshape(1, -1)])
            F = np.append(F, fx)
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return x, fx

        # --- Fit quadratic model from pool ---
        def fit_model(x0, max_points=40):
            nonlocal X, F, c_model, g_model, H_model, model_ok
            # Select points near x0
            if len(X) == 0:
                model_ok = False
                return
            d = np.linalg.norm(X - x0, axis=1)
            order = np.argsort(d)
            # Take up to max_points closest, but ensure enough for fitting
            m_needed = n * (n + 3) // 2  # number of quadratic parameters
            m_sel = min(len(X), max(max_points, m_needed + 5))
            idx = order[:m_sel]
            Xs = X[idx]
            Fs = F[idx]

            # Build design matrix for quadratic: [1, (X-x0), (X-x0)⊙(X-x0)]
            dx = Xs - x0
            # Symmetric quadratic terms
            quad_terms = []
            for i in range(n):
                for j in range(i, n):
                    quad_terms.append(dx[:, i] * dx[:, j])
            # Design matrix: 1 | dx | quad
            A = np.hstack([np.ones((len(Xs), 1)), dx, np.column_stack(quad_terms)])
            b = Fs

            # Solve least squares
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                model_ok = False
                return

            # Check condition: if rank is low, model may be bad
            if rank < A.shape[1] - 2:  # allow some rank deficiency
                model_ok = False
                return

            c_model = coeffs[0]
            g_model = coeffs[1:1 + n].copy()
            # Reconstruct Hessian from quadratic coefficients
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
            # Symmetrize
            H = 0.5 * (H + H.T)
            # Regularize if ill-conditioned
            eigvals, eigvecs = np.linalg.eigh(H)
            min_eig = np.min(np.abs(eigvals))
            if min_eig < 1e-8:
                H = H + 1e-6 * np.eye(n)
            H_model = H
            g_model = g_model
            model_ok = True

        # --- Solve trust-region subproblem: min q(x) s.t. ||x - x0|| <= rho ---
        def solve_tr_subproblem(x0, rho):
            # Minimize q(x) = c + g·(x-x0) + 0.5 (x-x0)ᵀH(x-x0)
            # subject to ||x - x0|| <= rho
            # Use iterative method: start at x0, move in steepest descent or Newton direction
            # Constrained by trust region

            # Compute steepest descent direction
            g = g_model
            H = H_model

            # If H is positive definite, Newton direction is -H^{-1} g
            # Otherwise use gradient direction
            try:
                # Check if H is PD
                eigvals = np.linalg.eigvalsh(H)
                if np.min(eigvals) > 1e-8:
                    d_newton = -np.linalg.solve(H, g)
                else:
                    d_newton = None
            except np.linalg.LinAlgError:
                d_newton = None

            # Steepest descent direction
            norm_g = np.linalg.norm(g)
            if norm_g < 1e-12:
                return x0.copy(), 0.0  # At critical point

            d_sd = -g / norm_g  # unit steepest descent
            # Step along steepest descent
            step_sd = rho  # full trust region step

            # Newton step scaled to trust region
            if d_newton is not None:
                norm_dn = np.linalg.norm(d_newton)
                if norm_dn > 1e-12:
                    d_newton_scaled = d_newton * min(1.0, rho / norm_dn)
                    # Compare predicted decrease
                    q_sd = 0.5 * norm_g * step_sd  # approx decrease along SD
                    q_nt = 0.5 * g @ d_newton_scaled  # approx decrease along Newton
                    if q_nt < q_sd:
                        d_step = d_newton_scaled
                    else:
                        d_step = d_sd * step_sd
                else:
                    d_step = d_sd * step_sd
            else:
                d_step = d_sd * step_sd

            x_new = x0 + d_step
            # Project to trust region if needed
            norm_step = np.linalg.norm(x_new - x0)
            if norm_step > rho:
                x_new = x0 + (x_new - x0) * (rho / norm_step)
            return x_new, np.linalg.norm(x_new - x0)

        # --- Main loop ---
        # Initial point
        rng = np.random.default_rng(42)
        x_init = rng.uniform(-5, 5, n)
        x_init, f_init = evaluate(x_init)
        x_cur = x_init.copy()
        f_cur = f_init

        # Build initial interpolation set: sample around x_init
        # Use a small sphere of points to fit initial model
        init_points = 0
        init_radius = 1.0
        # Generate points on a small sphere
        for _ in range(min(20, budget - 1)):
            direction = rng.standard_normal(n)
            direction /= (np.linalg.norm(direction) + 1e-12)
            x_try = x_cur + init_radius * direction
            x_try, f_try = evaluate(x_try)
            init_points += 1

        # Fit initial model
        fit_model(x_cur)

        # Main optimization loop
        while len(F) < budget:
            # Refit model at current best point
            fit_model(x_best)

            if model_ok:
                # Solve trust region subproblem
                x_cand, step_norm = solve_tr_subproblem(x_best, rho)
            else:
                # Fallback: random step or gradient-free step
                # Use a simple random perturbation
                direction = rng.standard_normal(n)
                direction /= (np.linalg.norm(direction) + 1e-12)
                x_cand = x_best + step_scale * direction
                step_norm = step_scale

            # Evaluate candidate
            x_cand, f_cand = evaluate(x_cand)

            # Trust region update
            actual_decrease = f_best - f_cand  # Note: f_best updated in evaluate
            # Predicted decrease from model
            if model_ok:
                dx = x_cand - x_best
                pred_decrease = -(g_model @ dx + 0.5 * dx @ H_model @ dx)
            else:
                pred_decrease = 1.0  # Assume some decrease

            # Ratio of actual to predicted
            if pred_decrease > 1e-12:
                ratio = actual_decrease / pred_decrease
            else:
                ratio = 1.0 if actual_decrease > 0 else 0.0

            if ratio > 0.75:
                # Good step: expand trust region
                rho = min(rho * 2.0, rho_max)
                x_cur = x_cand.copy()
                f_cur = f_cand
            elif ratio > 0.25:
                # Acceptable: keep same trust region
                x_cur = x_cand.copy()
                f_cur = f_cand
            else:
                # Bad step: shrink trust region
                rho = max(rho * 0.5, rho_min)
                # Don't move x_cur

            # Adapt step scale for fallback
            if not model_ok:
                if f_cand < f_best:
                    step_scale = min(step_scale * 1.5, 5.0)
                else:
                    step_scale = max(step_scale * 0.7, 0.01)

            # If stuck, add more exploration points
            if len(F) < budget and rho < 0.1:
                # Add a few random points to help refit
                for _ in range(min(5, budget - len(F))):
                    direction = rng.standard_normal(n)
                    direction /= (np.linalg.norm(direction) + 1e-12)
                    x_try = x_best + rho * 2 * direction
                    x_try, f_try = evaluate(x_try)

        return x_best, f_best

# EVOLVE-BLOCK-END
