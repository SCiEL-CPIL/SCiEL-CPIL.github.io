"""SQP-style optimizer with finite-difference gradients and Hessians for BBOB f12."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """SQP-style optimizer exploiting smooth convex-like structure of bent cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        deadline = t0 + 275.0  # safety margin vs. 330 s kill
        max_calls = budget

        calls = [0]
        best_fx = float('inf')
        best_x = np.zeros(dim)

        def f(x):
            nonlocal best_fx, best_x
            if calls[0] >= max_calls:
                return 1e18
            calls[0] += 1
            val = float(func(x))
            if not np.isfinite(val):
                return 1e18
            if val < best_fx:
                best_fx = val
                best_x = x.copy()
            return val

        def remaining():
            return deadline - time.time()

        def finite_diff_gradient(x, h=1e-5):
            """Estimate gradient using central differences."""
            g = np.zeros(dim)
            for i in range(dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] += h
                xm[i] -= h
                g[i] = (f(xp) - f(xm)) / (2 * h)
            return g

        def finite_diff_hessian(x, h=1e-4):
            """Estimate Hessian using central differences."""
            H = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(i, dim):
                    xpp = x.copy()
                    xpm = x.copy()
                    xmp = x.copy()
                    xmm = x.copy()
                    xpp[i] += h
                    xpp[j] += h
                    xpm[i] += h
                    xpm[j] -= h
                    xmp[i] -= h
                    xmp[j] += h
                    xmm[i] -= h
                    xmm[j] -= h
                    H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h * h)
                    H[j, i] = H[i, j]
            return H

        def sqp_step(x, fx, g, H, max_iter=20):
            """
            Sequential Quadratic Programming step:
            Solve QP subproblem: min g^T d + 0.5 d^T H d
            subject to ||x + d||_inf <= 5 (box constraints)
            Use simple projected gradient or solve the unconstrained QP then project.
            """
            n = dim
            # Solve (H + lambda*I) d = -g for stability
            # Use a small regularization to ensure positive definiteness
            reg = 1e-6
            H_reg = H + reg * np.eye(n)
            
            try:
                # Check if H_reg is positive definite; if not, use gradient direction
                eigvals = np.linalg.eigvalsh(H_reg)
                if np.min(eigvals) < 1e-8:
                    # Not positive definite, use steepest descent with line search
                    d = -g
                    d = d / (np.linalg.norm(d) + 1e-12)
                else:
                    d = np.linalg.solve(H_reg, -g)
            except np.linalg.LinAlgError:
                d = -g
                d = d / (np.linalg.norm(d) + 1e-12)

            # Limit step size to avoid large jumps initially
            max_step = 2.0
            step_norm = np.linalg.norm(d)
            if step_norm > max_step:
                d = d * (max_step / step_norm)

            # Line search with backtracking
            alpha = 1.0
            c1 = 1e-4
            x_new = x
            fx_new = fx
            success = False
            for _ in range(30):
                if calls[0] >= max_calls or remaining() < 5.0:
                    break
                x_trial = x + alpha * d
                x_trial = np.clip(x_trial, -5, 5)
                fx_trial = f(x_trial)
                
                # Armijo condition
                if fx_trial <= fx + c1 * alpha * (g @ d):
                    x_new = x_trial
                    fx_new = fx_trial
                    success = True
                    break
                alpha *= 0.5
            
            if not success or alpha < 1e-10:
                # If line search fails, try a smaller step in negative gradient direction
                d_alt = -g
                norm_g = np.linalg.norm(d_alt)
                if norm_g > 1e-12:
                    d_alt = d_alt / norm_g * 0.1
                    x_trial = x + d_alt
                    x_trial = np.clip(x_trial, -5, 5)
                    fx_trial = f(x_trial)
                    if fx_trial < fx:
                        x_new = x_trial
                        fx_new = fx_trial
                        success = True
            
            return x_new, fx_new, success

        def sqp_optimize(x0, fx0, max_iter=15):
            """Run SQP optimization starting from x0."""
            x = x0.copy()
            fx = fx0
            
            for _ in range(max_iter):
                if calls[0] >= max_calls or remaining() < 10.0:
                    break
                
                # Compute gradient
                g = finite_diff_gradient(x)
                
                # Check for convergence
                if np.linalg.norm(g) < 1e-7:
                    break
                
                # Compute Hessian (expensive but valuable for quadratic convergence)
                H = finite_diff_hessian(x)
                
                # Perform SQP step
                x_new, fx_new, success = sqp_step(x, fx, g, H)
                
                if not success:
                    break
                
                # Check improvement
                if fx_new >= fx - 1e-10:
                    # No improvement, try a smaller step
                    break
                
                x = x_new
                fx = fx_new
            
            return x, fx

        # Multi-start with deterministic and random points
        rng = np.random.default_rng(0)
        starts = []
        starts.append(np.zeros(dim))
        starts.append(rng.uniform(-4, 4, dim))
        starts.append(rng.uniform(-4, 4, dim))
        starts.append(np.array([3.0] * dim))
        starts.append(np.array([-3.0] * dim))

        # Allocate budget: more for first start (likely near optimum), less for others
        n_starts = len(starts)
        per_start = max(30, budget // n_starts)

        for si, s in enumerate(starts):
            if calls[0] >= max_calls or remaining() < 15.0:
                break
            
            s = np.clip(s, -5, 5)
            fx = f(s)
            
            # Run SQP
            x, fx = sqp_optimize(s, fx, max_iter=12)
            
            # Update best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Final polish: if we have budget left, do one more refinement from best point
        if calls[0] < max_calls and remaining() > 20.0:
            fx = best_fx
            x = best_x.copy()
            for _ in range(3):
                if calls[0] >= max_calls or remaining() < 10.0:
                    break
                g = finite_diff_gradient(x)
                if np.linalg.norm(g) < 1e-7:
                    break
                H = finite_diff_hessian(x)
                x, fx, success = sqp_step(x, fx, g, H)
                if not success:
                    break
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        if best_x is None or not np.all(np.isfinite(best_x)):
            best_x = np.zeros(dim)
            if calls[0] < max_calls:
                best_fx = f(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
