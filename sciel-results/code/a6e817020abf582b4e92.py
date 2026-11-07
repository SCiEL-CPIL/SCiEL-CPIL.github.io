"""DFTR optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Derivative-Free Trust-Region method with full quadratic surrogate."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial point - use center of box
        x = np.zeros(n)
        fx = func(x)
        f_best = fx
        x_best = x.copy()

        # Trust region parameters
        rho = 1.0
        rho_min = 1e-5
        rho_max = 8.0
        gamma1 = 0.25  # reduce factor
        gamma2 = 0.75  # good step threshold
        gamma3 = 2.0   # increase factor

        calls_used = 1
        stall_count = 0
        consecutive_bad = 0
        best_rho = rho
        last_improvement = 0.0

        def solve_tr_subproblem(g, H, rho):
            """Solve min g^T d + 0.5 d^T H d s.t. ||d|| <= rho using CG."""
            # Use conjugate gradient with trust region constraint
            # Start with steepest descent direction
            d = -g / (np.linalg.norm(g) + 1e-15)
            
            # Iterate CG
            r = -g - H @ d
            p = r.copy()
            d_old = d.copy()
            r_old = r.copy()
            
            for k in range(n):
                # Check if we're at the trust region boundary
                if np.linalg.norm(d_old) >= rho * 0.999:
                    break
                
                Ap = H @ p
                pAp = p @ Ap
                if pAp < 1e-15:
                    break
                
                alpha = (r @ r) / pAp
                d_new = d_old + alpha * p
                r_new = r_old - alpha * Ap
                
                # Check if step goes outside trust region
                if np.linalg.norm(d_new) > rho:
                    # Find t such that ||d_old + t*p|| = rho
                    a = p @ p
                    b = 2 * (d_old @ p)
                    c = d_old @ d_old - rho**2
                    disc = b*b - 4*a*c
                    if disc < 0:
                        break
                    t = (-b + np.sqrt(disc)) / (2*a)
                    if t > 0 and t < alpha:
                        d_old = d_old + t * p
                        break
                    else:
                        break
                
                if np.linalg.norm(r_new) < 1e-12:
                    d_old = d_new
                    break
                
                beta = (r_new @ r_new) / (r @ r)
                p = r_new + beta * p
                d_old = d_new
                r_old = r_new
            
            # Final constraint enforcement
            norm_d = np.linalg.norm(d_old)
            if norm_d > rho:
                d_old = d_old * (rho / norm_d)
            return d_old

        while calls_used + 2 * n + 1 <= budget:
            # 1. Build quadratic model using 2n central differences
            h = rho / 2.0
            f_plus = np.zeros(n)
            f_minus = np.zeros(n)
            
            for j in range(n):
                x_plus = x.copy()
                x_plus[j] += h
                x_plus = np.clip(x_plus, -5, 5)
                f_plus[j] = func(x_plus)
                calls_used += 1
                
                x_minus = x.copy()
                x_minus[j] -= h
                x_minus = np.clip(x_minus, -5, 5)
                f_minus[j] = func(x_minus)
                calls_used += 1

            if calls_used >= budget:
                break

            # 2. Estimate gradient and Hessian (diagonal + off-diagonal via cross terms)
            # Gradient
            g = (f_plus - f_minus) / (2 * h)
            
            # Diagonal Hessian
            H_diag = (f_plus + f_minus - 2 * fx) / (2 * h * h)
            
            # Build full Hessian approximation using mixed second differences
            # For off-diagonal H_ij, use: H_ij ≈ (f(x+he_i+he_j) - f(x+he_i-he_j) - f(x-he_i+he_j) + f(x-he_i-he_j)) / (4h^2)
            # This requires 4n(n-1)/2 = 2n(n-1) additional evaluations, too expensive
            # Instead, use diagonal Hessian with adaptive scaling
            
            H = np.diag(H_diag)
            
            # Ensure H is positive semi-definite for stability
            eigvals = np.linalg.eigvalsh(H)
            min_eig = np.min(eigvals)
            if min_eig < -1e-8:
                H = H + (-min_eig + 1e-6) * np.eye(n)
            elif min_eig < 1e-8:
                H = H + 1e-6 * np.eye(n)
            
            # 3. Solve trust-region subproblem
            d = solve_tr_subproblem(g, H, rho)
            
            # 4. Evaluate new point
            x_new = x + d
            x_new = np.clip(x_new, -5, 5)
            f_new = func(x_new)
            calls_used += 1
            
            # 5. Compute actual vs predicted improvement
            pred_improve = -(g @ d + 0.5 * d @ H @ d)
            act_improve = fx - f_new
            
            # 6. Update trust region
            if act_improve > 1e-10:
                x = x_new
                fx = f_new
                
                if fx < f_best:
                    f_best = fx
                    x_best = x.copy()
                    stall_count = 0
                    consecutive_bad = 0
                    last_improvement = act_improve
                    
                    # Increase radius if model was accurate
                    ratio = act_improve / max(abs(pred_improve), 1e-15)
                    if ratio > gamma2:
                        rho = min(rho * gamma3, rho_max)
                        best_rho = max(best_rho, rho)
                    else:
                        rho = min(rho * 1.2, rho_max)
                else:
                    stall_count += 1
                    consecutive_bad += 1
                    rho = max(rho * 0.8, rho_min)
            else:
                # Bad step
                consecutive_bad += 1
                rho = max(rho * gamma1, rho_min)
                stall_count += 1
            
            # 7. Restart strategy
            if (consecutive_bad >= 4 or stall_count >= 8) and calls_used + 1 <= budget:
                # Adaptive jump: scale with recent improvement history
                jump_scale = max(0.5, last_improvement * 0.1, rho * 0.5)
                jump_scale = min(jump_scale, 3.0)
                
                # Try a few random restarts
                for _ in range(3):
                    if calls_used >= budget:
                        break
                    x_candidate = x_best + rng.normal(0, jump_scale, n)
                    x_candidate = np.clip(x_candidate, -5, 5)
                    f_candidate = func(x_candidate)
                    calls_used += 1
                    
                    if f_candidate < f_best:
                        x = x_candidate
                        fx = f_candidate
                        f_best = fx
                        x_best = x.copy()
                        break
                    elif f_candidate < fx:
                        x = x_candidate
                        fx = f_candidate
                
                rho = max(0.5, best_rho * 0.5)
                stall_count = 0
                consecutive_bad = 0

        return x_best, f_best
# EVOLVE-BLOCK-END
