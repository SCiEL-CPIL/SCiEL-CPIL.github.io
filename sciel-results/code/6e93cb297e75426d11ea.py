"""Hybrid Nelder-Mead + Hessian Newton + (1+1)-ES optimizer for BBOB f11 Discus."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex, Hessian Newton polish, then (1+1)-ES for BBOB-style Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def estimate_gradient(x_center, step=0.01):
            """Estimate gradient via central finite differences. Requires 2*dim evals."""
            nonlocal call_count, best_f, best_x
            if call_count + 2 * dim > budget:
                return None
            grad = np.zeros(dim)
            for i in range(dim):
                xp = x_center.copy()
                xp[i] += step
                xmp = x_center.copy()
                xmp[i] -= step
                fp = try_eval(xp)
                if fp is None:
                    return None
                fmp = try_eval(xmp)
                if fmp is None:
                    return None
                grad[i] = (fp - fmp) / (2 * step)
            return grad

        def estimate_hessian_diagonal(x_center, step=0.05):
            """Estimate diagonal of Hessian via central second differences. Requires dim evals beyond gradient."""
            nonlocal call_count, best_f, best_x
            if call_count + dim > budget:
                return None
            H_diag = np.zeros(dim)
            for i in range(dim):
                # We already have f(x+step*e_i) and f(x-step*e_i) from gradient, but let's use larger step
                # Use a separate larger step for curvature estimation
                sp = step
                xp = x_center.copy()
                xp[i] += sp
                xmp = x_center.copy()
                xmp[i] -= sp
                fc = try_eval(x_center)
                if fc is None:
                    return None
                fp = try_eval(xp)
                if fp is None:
                    return None
                fmp = try_eval(xmp)
                if fmp is None:
                    return None
                H_diag[i] = (fp - 2 * fc + fmp) / (sp * sp)
            return H_diag

        def newton_step_diagonal(x, H_diag, grad, max_iter=10):
            """Apply damped Newton steps using diagonal Hessian approximation.
            x_new[i] = x[i] - grad[i] / H_diag[i]
            Returns (x_new, f_new)."""
            nonlocal call_count, best_f, best_x
            x = x.copy()
            fx = best_f
            
            # Regularize: ensure positive diagonal entries
            H_safe = np.where(H_diag > 1e-8, H_diag, 1e-8)
            
            for _ in range(max_iter):
                if call_count >= budget:
                    break
                if time.time() > deadline - 10:
                    break
                
                # Newton direction with diagonal Hessian
                direction = -grad / H_safe
                # Damp the step
                x_new = x + direction
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = try_eval(x_new)
                if f_new is None:
                    break
                
                if f_new < fx:
                    x = x_new
                    fx = f_new
                    # Re-estimate gradient
                    grad_new = estimate_gradient(x, step=0.005)
                    if grad_new is None:
                        break
                    grad = grad_new
                    if np.linalg.norm(grad) < 1e-10:
                        break
                else:
                    # Step rejected, try smaller step
                    x_new = x + 0.5 * direction
                    x_new = np.clip(x_new, -5.0, 5.0)
                    f_new = try_eval(x_new)
                    if f_new is None:
                        break
                    if f_new < fx:
                        x = x_new
                        fx = f_new
                        grad_new = estimate_gradient(x, step=0.005)
                        if grad_new is None:
                            break
                        grad = grad_new
                        if np.linalg.norm(grad) < 1e-10:
                            break
                    else:
                        break
            
            return x, fx

        # Phase 1: Nelder-Mead Simplex Method
        x0 = np.zeros(dim)
        f0 = try_eval(x0)
        if f0 is None:
            return best_x, best_f

        simplex = [x0.copy()]
        simplex_f = [f0]
        
        for i in range(dim):
            xi = x0.copy()
            xi[i] += 0.5
            fi = try_eval(xi)
            if fi is None:
                break
            simplex.append(xi)
            simplex_f.append(fi)
        
        if len(simplex) >= dim + 1:
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5
            
            nm_budget = max(50, budget - call_count - 80)
            no_improve = 0
            for _ in range(nm_budget):
                if call_count >= budget:
                    break
                if time.time() > deadline - 30:
                    break
                
                sorted_idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in sorted_idx]
                simplex_f = [simplex_f[i] for i in sorted_idx]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = try_eval(xr)
                if fr is None:
                    break
                
                improved = False
                if fr < simplex_f[0]:
                    if fr < simplex_f[-2]:
                        xe = centroid + gamma * (centroid - simplex[-1])
                        fe = try_eval(xe)
                        if fe is None:
                            break
                        if fe < fr:
                            simplex[-1] = xe
                            simplex_f[-1] = fe
                        else:
                            simplex[-1] = xr
                            simplex_f[-1] = fr
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    improved = True
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                    improved = True
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = try_eval(xc)
                    if fc is None:
                        break
                    if fc < fr:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                        improved = True
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            fi = try_eval(simplex[i])
                            if fi is None:
                                break
                            simplex_f[i] = fi
                        if call_count >= budget:
                            break
                
                if improved:
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= 10:
                    break

        # Phase 2: Diagonal Hessian Newton polish
        if best_x is not None and call_count < budget - 20:
            grad = estimate_gradient(best_x.copy(), step=0.01)
            if grad is not None:
                H_diag = estimate_hessian_diagonal(best_x.copy(), step=0.05)
                if H_diag is not None:
                    x_new, f_new = newton_step_diagonal(best_x.copy(), H_diag, grad, max_iter=10)

        # Phase 3: (1+1)-ES polish with adaptive sigma
        if best_x is not None and call_count < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(budget - call_count):
                if time.time() > deadline - 5:
                    break
                if call_count >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.85

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
