import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def initial_grid(n):
        """Create initial grid placement with slightly smaller radii for feasibility."""
        side = int(np.ceil(np.sqrt(n)))
        xs = np.zeros(n)
        ys = np.zeros(n)
        rs = np.zeros(n)
        for i in range(n):
            row = i // side
            col = i % side
            xs[i] = (col + 0.5) / side
            ys[i] = (row + 0.5) / side
            rs[i] = 0.5 / side - 1e-4
        return xs, ys, rs

    def check_validity(xs, ys, rs, tol=1e-9):
        """Check if circles are valid (inside square, non-overlapping)."""
        for i in range(N):
            if xs[i] < rs[i] - tol or xs[i] > 1.0 - rs[i] + tol:
                return False
            if ys[i] < rs[i] - tol or ys[i] > 1.0 - rs[i] + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                dist_sq = dx * dx + dy * dy
                min_dist = rs[i] + rs[j]
                if dist_sq < min_dist * min_dist - tol * (min_dist * min_dist):
                    return False
        return True

    def objective_and_grad(params, beta):
        """
        Compute objective function and gradient.
        params: [x_0...x_n-1, y_0...y_n-1, r_0...r_n-1]
        beta: penalty weight for soft constraints.
        
        Objective: -sum(r) + beta * (soft_plus(violations))
        """
        xs = params[:N]
        ys = params[N:2*N]
        rs = params[2*N:3*N]
        
        # 1. Main term: maximize sum of radii -> minimize -sum(r)
        obj = -np.sum(rs)
        grad = np.zeros(3*N)
        grad[2*N:3*N] = -1.0
        
        # 2. Penalty term: smooth approximation of max(0, violation)
        # We use softplus(v) = log(1 + exp(v)) / alpha as a smooth max(0, v)
        # Here we accumulate violations into a vector and use log-sum-exp for stability
        
        # Boundary violations
        # x < r -> x - r < 0 -> violation is r - x
        # x > 1-r -> 1 - r - x < 0 -> violation is x - (1-r)
        v_left = rs - xs
        v_right = xs - (1.0 - rs)
        v_bottom = rs - ys
        v_top = ys - (1.0 - rs)
        
        # Overlap violations
        # dist < r_i + r_j -> violation is (r_i + r_j) - dist
        overlap_violations = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-8:
                    dist = 1e-8
                v = rs[i] + rs[j] - dist
                overlap_violations.append(v)
        
        all_violations = np.concatenate([v_left, v_right, v_bottom, v_top, np.array(overlap_violations)])
        
        # Soft plus approximation: log(1 + exp(v))
        # To avoid overflow, use max(0, v) + log(1 + exp(-|v|))
        # Actually, standard softplus is log(1+exp(x)).
        # We want a smooth max(0, x). 
        # softplus(x) - log(2) approximates max(0, x) for large |x|.
        # Let's use: penalty = (1/beta) * sum(softplus(beta * v))
        # This converges to sum(max(0, v)) as beta -> inf.
        
        # Gradient of (1/beta) * softplus(beta * v) wrt v is sigmoid(beta * v)
        # So we can compute gradients directly.
        
        # To be numerically stable, we compute the penalty value and gradient.
        # penalty_val = sum( log(1 + exp(beta * v_i)) ) / beta
        # grad_v = sigmoid(beta * v_i)
        
        # We accumulate gradients for each parameter.
        
        # Helper to add gradient contribution
        def add_grad(param_idx, v_val, v_grad):
            # v_grad is sigmoid(beta * v)
            # Chain rule: d(obj)/d(param) += beta * v_grad * d(v)/d(param) / beta = v_grad * d(v)/d(param)
            # Wait, the term is (1/beta) * softplus(beta * v).
            # d/dv = (1/beta) * beta * sigmoid(beta*v) = sigmoid(beta*v).
            # So the gradient contribution to v is sigmoid(beta*v).
            # Then we multiply by d(v)/d(param).
            pass

        # Let's just compute the scalar penalty and the gradient vector manually for clarity.
        
        # Precompute sigmoid(beta * v) for all violations
        # We need to be careful with large beta * v
        # sigmoid(x) = 1 / (1 + exp(-x))
        # If x > 50, sigmoid ~ 1
        # If x < -50, sigmoid ~ 0
        
        sig_vals = np.zeros_like(all_violations)
        for k, v in enumerate(all_violations):
            bv = beta * v
            if bv > 50:
                sig_vals[k] = 1.0
            elif bv < -50:
                sig_vals[k] = 0.0
            else:
                sig_vals[k] = 1.0 / (1.0 + np.exp(-bv))
        
        # Penalty Value
        # softplus(x) = log(1+exp(x))
        # For stability:
        def softplus(x):
            if x > 50:
                return x
            elif x < -50:
                return 0.0
            else:
                return np.log(1.0 + np.exp(x))
        
        penalty_val = 0.0
        for k, v in enumerate(all_violations):
            penalty_val += softplus(beta * v)
        penalty_val /= beta
        
        obj += penalty_val
        
        # Gradients
        # d(penalty)/d(param) = sum_k sigmoid(beta * v_k) * d(v_k)/d(param)
        
        # Reset grad for penalty part (we already have -1 for radii)
        # Actually, let's just add to the existing grad.
        
        # Boundary Gradients
        # v_left = r_i - x_i
        # d(v_left)/dx_i = -1, d(v_left)/dr_i = 1
        for i in range(N):
            s = sig_vals[i] # v_left
            grad[i] += s * (-1.0)
            grad[2*N + i] += s * (1.0)
            
            s = sig_vals[N + i] # v_right
            grad[i] += s * (1.0)
            grad[2*N + i] += s * (-1.0)
            
            s = sig_vals[2*N + i] # v_bottom
            grad[N + i] += s * (-1.0)
            grad[2*N + i] += s * (1.0)
            
            s = sig_vals[3*N + i] # v_top
            grad[N + i] += s * (1.0)
            grad[2*N + i] += s * (-1.0)
        
        # Overlap Gradients
        # v_ij = r_i + r_j - dist_ij
        # d(v_ij)/dx_i = -(x_j - x_i)/dist
        # d(v_ij)/dx_j = (x_j - x_i)/dist
        # d(v_ij)/dy_i = -(y_j - y_i)/dist
        # d(v_ij)/dy_j = (y_j - y_i)/dist
        # d(v_ij)/dr_i = 1
        # d(v_ij)/dr_j = 1
        
        idx = 4*N # Start index for overlap violations in all_violations
        for i in range(N):
            for j in range(i + 1, N):
                s = sig_vals[idx]
                idx += 1
                if s == 0.0:
                    continue
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-8:
                    dist = 1e-8
                ux = dx / dist
                uy = dy / dist
                
                # grad wrt x_i
                grad[i] += s * (-ux)
                # grad wrt x_j
                grad[j] += s * (ux)
                # grad wrt y_i
                grad[N + i] += s * (-uy)
                # grad wrt y_j
                grad[N + j] += s * (uy)
                # grad wrt r_i
                grad[2*N + i] += s * (1.0)
                # grad wrt r_j
                grad[2*N + j] += s * (1.0)
        
        return obj, grad

    # --- Main Optimization Loop ---

    # Phase 1: Initial Grid
    xs, ys, rs = initial_grid(N)
    params = np.concatenate([xs, ys, rs])
    
    best_score = np.sum(rs)
    best_params = params.copy()
    
    # Phase 2: Progressive Beta Optimization
    # Start with low beta (smooth landscape), increase beta (sharp constraints)
    betas = [10, 50, 100, 500, 1000, 5000, 10000]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        # Use current best params as start
        current_params = best_params.copy()
        
        # Run L-BFGS-B
        # Bounds:
        # x, y in [0, 1]
        # r in [0, 0.5]
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
        
        try:
            res = minimize(
                lambda p: objective_and_grad(p, beta)[0],
                current_params,
                jac=lambda p: objective_and_grad(p, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < objective_and_grad(current_params, beta)[0]:
                # Check validity of the result
                p = res.x
                xs_new = p[:N]
                ys_new = p[N:2*N]
                rs_new = p[2*N:3*N]
                
                if check_validity(xs_new, ys_new, rs_new):
                    score = np.sum(rs_new)
                    if score > best_score:
                        best_score = score
                        best_params = p.copy()
                else:
                    # If invalid, we might try to repair or just keep previous
                    # For now, if it's slightly invalid, we can try a few steps of PBD repair
                    # But let's stick to the differentiable approach.
                    # If L-BFGS-B ends up invalid, it means the penalty wasn't strong enough
                    # or it got stuck. We'll keep the previous best valid state.
                    pass
                    
        except Exception as e:
            pass

    # Phase 3: High Precision Polish with very high beta
    if time.time() < deadline - 5:
        final_beta = 10000
        current_params = best_params.copy()
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
        try:
            res = minimize(
                lambda p: objective_and_grad(p, final_beta)[0],
                current_params,
                jac=lambda p: objective_and_grad(p, final_beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            p = res.x
            xs_new = p[:N]
            ys_new = p[N:2*N]
            rs_new = p[2*N:3*N]
            
            if check_validity(xs_new, ys_new, rs_new):
                score = np.sum(rs_new)
                if score > best_score:
                    best_score = score
                    best_params = p.copy()
        except:
            pass

    # Final Validation
    best_xs = best_params[:N]
    best_ys = best_params[N:2*N]
    best_rs = best_params[2*N:3*N]
    
    if not check_validity(best_xs, best_ys, best_rs):
        # Fallback: return a valid grid
        xs_f, ys_f, rs_f = initial_grid(N)
        best_xs, best_ys, best_rs = xs_f, ys_f, rs_f

    return np.column_stack((best_xs, best_ys, best_rs))
