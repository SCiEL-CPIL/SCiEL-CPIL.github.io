"""Circle packing in perimeter-4 rectangle using softplus-penalty L-BFGS-B with multi-start hexagonal lattices."""
import numpy as np
import time
from scipy.optimize import minimize
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def hexagonal_grid(w, h, n_cols, n_rows, offset_rows=None):
        """
        Generate a hexagonal (offset) grid of circle centers.
        Returns arrays x, y of shape (n_cols*n_rows,).
        """
        centers = []
        for row in range(n_rows):
            for col in range(n_cols):
                # Hexagonal offset: shift odd rows by half the horizontal spacing
                if offset_rows is not None and row in offset_rows:
                    cx = (col + 0.5) * (w / n_cols)
                else:
                    cx = (col + 0.5) * (w / n_cols)
                cy = (row + 0.5) * (h / n_rows)
                centers.append((cx, cy))
        return np.array(centers).flatten()
    
    def make_initial_config(w, layout):
        """
        Create an initial configuration given width w and a layout specification.
        layout is a tuple (n_cols, n_rows, offset_rows_set).
        Returns x, y, r arrays.
        """
        h = 2.0 - w
        n_cols, n_rows, offset_rows = layout
        total = n_cols * n_rows
        if total != N:
            return None
        
        x = np.zeros(N)
        y = np.zeros(N)
        
        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            if offset_rows is not None and row in offset_rows:
                x[i] = (col + 0.5) * (w / n_cols)
            else:
                x[i] = (col + 0.5) * (w / n_cols)
            y[i] = (row + 0.5) * (h / n_rows)
        
        # Initial radii: start small to allow movement
        r = np.full(N, 0.05)
        
        return x, y, r
    
    def objective_and_grad(params, beta=100.0):
        """
        Softplus penalty objective: minimize -sum(r) + beta * sum(softplus(violations)).
        params = [w, x_0, ..., x_{N-1}, y_0, ..., y_{N-1}, r_0, ..., r_{N-1}]
        Returns (value, gradient).
        """
        w = params[0]
        x = params[1:1+N]
        y = params[1+N:1+2*N]
        r = params[1+2*N:1+3*N]
        
        h = 2.0 - w
        
        # Sum of radii (we want to maximize, so minimize negative)
        sum_r = np.sum(r)
        
        # Violations:
        # 1. Boundary violations
        # 2. Overlap violations
        
        violations = []
        grad_w = 0.0
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        grad_r = np.zeros(N)
        
        # Boundary violations
        for i in range(N):
            # Left boundary: x[i] - r[i] >= 0
            viol_left = -(x[i] - r[i])  # negative means violation
            if viol_left > 0:
                sp = np.log1p(np.exp(beta * viol_left)) / beta
                dsdviol = np.exp(beta * viol_left) / (1 + np.exp(beta * viol_left))
                grad_x[i] += beta * sp * (-1) * dsdviol  # d(-sp)/dx
                grad_r[i] += beta * sp * (1) * dsdviol   # d(-sp)/dr
                grad_w += 0  # w doesn't directly affect left boundary in this form
            
            # Right boundary: x[i] + r[i] <= w
            viol_right = (x[i] + r[i]) - w
            if viol_right > 0:
                sp = np.log1p(np.exp(beta * viol_right)) / beta
                dsdviol = np.exp(beta * viol_right) / (1 + np.exp(beta * viol_right))
                grad_x[i] += beta * sp * (1) * dsdviol
                grad_r[i] += beta * sp * (1) * dsdviol
                grad_w += beta * sp * (-1) * dsdviol
            
            # Bottom boundary: y[i] - r[i] >= 0
            viol_bottom = -(y[i] - r[i])
            if viol_bottom > 0:
                sp = np.log1p(np.exp(beta * viol_bottom)) / beta
                dsdviol = np.exp(beta * viol_bottom) / (1 + np.exp(beta * viol_bottom))
                grad_y[i] += beta * sp * (-1) * dsdviol
                grad_r[i] += beta * sp * (1) * dsdviol
            
            # Top boundary: y[i] + r[i] <= h = 2 - w
            viol_top = (y[i] + r[i]) - (2.0 - w)
            if viol_top > 0:
                sp = np.log1p(np.exp(beta * viol_top)) / beta
                dsdviol = np.exp(beta * viol_top) / (1 + np.exp(beta * viol_top))
                grad_y[i] += beta * sp * (1) * dsdviol
                grad_r[i] += beta * sp * (1) * dsdviol
                grad_w += beta * sp * (1) * dsdviol  # d/dw of (2-w) is -1, so violation increases with w
        
        # Overlap violations: for each pair (i,j), dist(i,j) >= r[i] + r[j]
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + 1e-12)
                min_dist = r[i] + r[j]
                viol = min_dist - dist
                if viol > 0:
                    sp = np.log1p(np.exp(beta * viol)) / beta
                    dsdviol = np.exp(beta * viol) / (1 + np.exp(beta * viol))
                    # d(viol)/dx[i] = -dx/dist, d(viol)/dx[j] = dx/dist
                    # d(viol)/dy[i] = -dy/dist, d(viol)/dy[j] = dy/dist
                    # d(viol)/dr[i] = 1, d(viol)/dr[j] = 1
                    factor = beta * sp * dsdviol
                    grad_x[i] += factor * (-dx / dist)
                    grad_x[j] += factor * (dx / dist)
                    grad_y[i] += factor * (-dy / dist)
                    grad_y[j] += factor * (dy / dist)
                    grad_r[i] += factor * 1
                    grad_r[j] += factor * 1
        
        # Total objective: -sum_r + beta * sum(sp(violations))
        # We already accumulated beta*sp*dsdviol in gradients, so the objective value is:
        obj_val = -sum_r
        
        # Add the softplus penalties to the objective value
        for i in range(N):
            viol_left = -(x[i] - r[i])
            if viol_left > 0:
                obj_val += np.log1p(np.exp(beta * viol_left)) / beta
            
            viol_right = (x[i] + r[i]) - w
            if viol_right > 0:
                obj_val += np.log1p(np.exp(beta * viol_right)) / beta
            
            viol_bottom = -(y[i] - r[i])
            if viol_bottom > 0:
                obj_val += np.log1p(np.exp(beta * viol_bottom)) / beta
            
            viol_top = (y[i] + r[i]) - (2.0 - w)
            if viol_top > 0:
                obj_val += np.log1p(np.exp(beta * viol_top)) / beta
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + 1e-12)
                min_dist = r[i] + r[j]
                viol = min_dist - dist
                if viol > 0:
                    obj_val += np.log1p(np.exp(beta * viol)) / beta
        
        # Gradient
        grad = np.empty(1 + 3*N)
        grad[0] = grad_w
        grad[1:1+N] = grad_x
        grad[1+N:1+2*N] = grad_y
        grad[1+2*N:1+3*N] = grad_r
        
        return obj_val, grad
    
    def check_validity(w, x, y, r):
        """Strict validity check."""
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return False
        
        for i in range(N):
            if r[i] < 1e-9:
                return False
            if x[i] - r[i] < -1e-8 or x[i] + r[i] > w + 1e-8:
                return False
            if y[i] - r[i] < -1e-8 or y[i] + r[i] > h + 1e-8:
                return False
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - 1e-9)**2:
                    return False
        return True
    
    def polish(w, x, y, r):
        """Greedy radius increase polish."""
        for _ in range(100):
            improved = False
            for i in range(N):
                old_r = r[i]
                new_r = old_r * 1.005
                r[i] = new_r
                if check_validity(w, x, y, r):
                    improved = True
                else:
                    r[i] = old_r
            if not improved:
                break
        return r
    
    # Multi-start configurations
    # Different aspect ratios and grid layouts
    configs = []
    
    # Layout 1: 7x3 grid
    for w in [1.2, 1.3, 1.4, 1.5, 1.6]:
        h = 2.0 - w
        if h > 0.1 and w > 0.1:
            x, y, r = make_initial_config(w, (7, 3, None))
            if x is not None:
                configs.append((w, x.copy(), y.copy(), r.copy()))
    
    # Layout 2: 6x4 grid (24 slots, use 21)
    # Actually 6*4=24, we need exactly 21. Let's use partial grids.
    # 7x3 = 21 exactly.
    # 6x3 = 18, not enough.
    # 7x3 is the main one.
    
    # Add some perturbed versions
    for w in [1.3, 1.4, 1.5]:
        h = 2.0 - w
        x, y, r = make_initial_config(w, (7, 3, None))
        if x is not None:
            # Add small random perturbation
            x_pert = x + np.random.randn(N) * 0.01
            y_pert = y + np.random.randn(N) * 0.01
            configs.append((w, x_pert, y_pert, r.copy()))
    
    best_value = -np.inf
    best_params = None
    
    n_configs = len(configs)
    time_per_config = (deadline - start_time) / max(n_configs, 1)
    
    for cfg_idx, (w, x, y, r) in enumerate(configs):
        if time.time() > deadline - 10:
            break
        
        # Build parameter vector
        params = np.concatenate([[w], x, y, r])
        
        # Bounds
        bounds = [(0.2, 1.8)]  # w
        for i in range(N):
            bounds.append((0, 2.0))  # x[i]
        for i in range(N):
            bounds.append((0, 2.0))  # y[i]
        for i in range(N):
            bounds.append((0.01, 0.5))  # r[i]
        
        # Run L-BFGS-B with increasing beta for annealing-like effect
        beta_schedule = [50, 100, 200, 500]
        
        for beta in beta_schedule:
            if time.time() > deadline - 10:
                break
            
            def obj_grad(p, b=beta):
                val, grad = objective_and_grad(p, b)
                return val, grad
            
            try:
                res = minimize(
                    obj_grad,
                    params,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                params = res.x
            except Exception:
                pass
        
        # Extract and check
        w_opt = params[0]
        x_opt = params[1:1+N]
        y_opt = params[1+N:1+2*N]
        r_opt = params[1+2*N:1+3*N]
        
        if check_validity(w_opt, x_opt, y_opt, r_opt):
            # Polish
            r_opt = polish(w_opt, x_opt, y_opt, r_opt)
            val = np.sum(r_opt)
            if val > best_value:
                best_value = val
                best_params = (w_opt, x_opt.copy(), y_opt.copy(), r_opt.copy())
        
        # Also try with the original grid positions but optimized radii
        # (in case L-BFGS moved things too much)
        if check_validity(w, x, y, r):
            r_polished = polish(w, x.copy(), y.copy(), r.copy())
            val = np.sum(r_polished)
            if val > best_value:
                best_value = val
                best_params = (w, x.copy(), y.copy(), r_polished.copy())
    
    # If no valid solution found, fall back to a simple valid configuration
    if best_params is None:
        w = 1.4
        h = 0.6
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.full(N, 0.08)
        for i in range(N):
            col = i % 7
            row = i // 7
            x[i] = (col + 0.5) * w / 7
            y[i] = (row + 0.5) * h / 3
        if not check_validity(w, x, y, r):
            r[:] = 0.07
            if not check_validity(w, x, y, r):
                r[:] = 0.06
        best_params = (w, x.copy(), y.copy(), r.copy())
    
    bw, bx, by, br = best_params
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [bx[i], by[i], br[i]]
        
    return result

# EVOLVE-BLOCK-END
