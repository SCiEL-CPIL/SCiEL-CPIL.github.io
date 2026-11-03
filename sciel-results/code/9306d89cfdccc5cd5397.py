"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _generate_hex_init(W, H, n_circles):
    """
    Generate an initial configuration using a hexagonal lattice, 
    scaled to fit the rectangle.
    """
    # Ideal hexagonal packing density is pi / (2 * sqrt(3)) ~ 0.9069
    # We want to place n_circles.
    # Let's estimate the required side length for a hex grid.
    # Area of rectangle = W * H.
    # Area per circle in hex packing ~ (2 * sqrt(3)) * r^2.
    # n * (2 * sqrt(3)) * r^2 <= W * H * packing_efficiency
    # Let's just try a few grid sizes and pick the one that fits best or use a simple heuristic.
    
    # Simple heuristic: 
    # rows * cols >= n_circles
    # Let's try to find a good aspect ratio for the grid.
    
    best_init = None
    best_score = -1
    
    # Try different grid orientations and scales
    # A hex grid can be represented by rows and columns.
    # Distance between centers horizontally = 2r
    # Distance between centers vertically = sqrt(3) r
    # Offset every other row by r
    
    # We will generate a set of points, then scale them to fit W, H.
    
    max_dim = max(W, H)
    min_dim = min(W, H)
    
    # Try grid sizes
    for rows in range(3, 8):
        cols = int(math.ceil(n_circles / rows))
        if cols < 1:
            continue
            
        # Generate hex points in a normalized space
        # Let r = 0.5 be the base radius for spacing calculation
        # x = col * 2r + (row % 2) * r
        # y = row * sqrt(3) * r
        
        r_base = 0.5
        pts = []
        for i in range(rows):
            for j in range(cols):
                x = j * 2 * r_base + (i % 2) * r_base
                y = i * math.sqrt(3) * r_base
                pts.append((x, y))
        
        if len(pts) < n_circles:
            continue
            
        # Take the first n_circles points (or centered subset?)
        # Let's take a centered subset to avoid boundary issues initially
        # Actually, just take the first N and center them
        
        pts = np.array(pts[:n_circles])
        
        # Center the points
        cx = pts[:, 0]
        cy = pts[:, 1]
        
        cx_min, cx_max = cx.min(), cx.max()
        cy_min, cy_max = cy.min(), cy.max()
        
        w_grid = cx_max - cx_min
        h_grid = cy_max - cy_min
        
        if w_grid < 1e-6 or h_grid < 1e-6:
             continue
            
        # Scale to fit W, H with some margin
        # We want the grid to fit inside [0, W] x [0, H]
        # Margin for radius. Let's assume r ~ min(W, H) / (2 * max(rows, cols))
        
        # Scale factor
        scale_x = W / w_grid
        scale_y = H / h_grid
        scale = min(scale_x, scale_y) * 0.95 # 5% margin
        
        cx_scaled = (cx - cx_min) * scale + (W - w_grid * scale) / 2
        cy_scaled = (cy - cy_min) * scale + (H - h_grid * scale) / 2
        
        # Estimate radius
        # In hex packing, min distance is 2r.
        # The scale factor maps the base spacing (2*r_base = 1) to 2*scale*r_base = scale.
        # So r ~ scale * r_base = scale * 0.5
        r_est = scale * 0.5 * 0.95 # Slightly smaller to ensure validity
        
        # Check validity and score
        valid = True
        sum_r = 0
        for i in range(n_circles):
            if cx_scaled[i] - r_est < 0 or cx_scaled[i] + r_est > W:
                valid = False
                break
            if cy_scaled[i] - r_est < 0 or cy_scaled[i] + r_est > H:
                valid = False
                break
            sum_r += r_est
        
        if not valid:
            continue
            
        # Quick overlap check
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                dx = cx_scaled[i] - cx_scaled[j]
                dy = cy_scaled[i] - cy_scaled[j]
                dist = math.hypot(dx, dy)
                if dist < 2 * r_est - 1e-6:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            if sum_r > best_score:
                best_score = sum_r
                best_init = np.column_stack((cx_scaled, cy_scaled, np.full(n_circles, r_est)))
                
    if best_init is None:
        # Fallback: Grid
        side = int(math.ceil(math.sqrt(n_circles)))
        cx = np.linspace(0.1, W-0.1, side)
        cy = np.linspace(0.1, H-0.1, side)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_flat = cx_grid.flatten()
        cy_flat = cy_grid.flatten()
        r_flat = np.full(n_circles, 0.1)
        best_init = np.column_stack((cx_flat[:n_circles], cy_flat[:n_circles], r_flat))
        
    return best_init

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_score = -1
    best_result = None
    
    # Perimeter 4 => W + H = 2
    # Try a few aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 / (1.0 + ar)
        
        # Initial guess from hex lattice
        x0 = _generate_hex_init(W, H, N)
        
        # SCR Loop
        # We will iterate:
        # 1. Solve QP: Minimize -sum(r) + 0.5 * sum((d_ij - (r_i + r_j))^2) subject to linearized constraints
        #    Actually, standard SCR for circle packing:
        #    Minimize 0.5 * ||x - x_k||^2 - mu * sum(r)
        #    Subject to:
        #       (x_i - x_j)^T (x_i - x_j) >= (r_i + r_j)^2  (Non-convex)
        #    Linearize the quadratic constraint at x_k:
        #       2 * (x_i_k - x_j_k)^T (x_i - x_j) >= (r_i + r_j)^2
        #       Note: (x_i - x_j) is variable.
        #       This is a linear constraint on x_i and x_j.
        
        #    Also boundary constraints:
        #       r_i <= x_i_x
        #       r_i <= W - x_i_x
        #       r_i <= x_i_y
        #       r_i <= H - x_i_y
        #       r_i >= 0
        
        #    Objective: Maximize sum(r). 
        #    To make it a minimization: Minimize -sum(r).
        #    To keep x close to previous iterate (trust region / regularization):
        #    Minimize 0.5 * ||x - x_k||^2 - lambda * sum(r)
        
        x_current = x0
        lambda_val = 1.0
        converged = False
        
        for it in range(50):
            if time.time() > deadline:
                break
                
            # Define objective for SLSQP
            # x = [cx, cy, r]
            def objective(x):
                cx = x[:N]
                cy = x[N:2*N]
                r = x[2*N:]
                
                # Regularization term
                diff_cx = cx - x_current[:N]
                diff_cy = cy - x_current[N:2*N]
                diff_r = r - x_current[2*N:]
                
                reg = 0.5 * (np.sum(diff_cx**2) + np.sum(diff_cy**2) + np.sum(diff_r**2))
                
                # Maximize sum(r) -> Minimize -sum(r)
                return reg - lambda_val * np.sum(r)
            
            # Define constraints
            # SLSQP constraints are of the form g(x) >= 0
            constraints = []
            
            # Boundary constraints
            # cx - r >= 0
            def c_left(x):
                return x[:N] - x[2*N:]
            constraints.append({'type': 'ineq', 'fun': c_left})
            
            # W - cx - r >= 0
            def c_right(x):
                return W - x[:N] - x[2*N:]
            constraints.append({'type': 'ineq', 'fun': c_right})
            
            # cy - r >= 0
            def c_bottom(x):
                return x[N:2*N] - x[2*N:]
            constraints.append({'type': 'ineq', 'fun': c_bottom})
            
            # H - cy - r >= 0
            def c_top(x):
                return H - x[N:2*N] - x[2*N:]
            constraints.append({'type': 'ineq', 'fun': c_top})
            
            # r >= 0
            def c_pos_r(x):
                return x[2*N:]
            constraints.append({'type': 'ineq', 'fun': c_pos_r})
            
            # Overlap constraints (Linearized)
            # For each pair i, j:
            # 2 * (cx_i_k - cx_j_k) * (cx_i - cx_j) + 2 * (cy_i_k - cy_j_k) * (cy_i - cy_j) >= (r_i + r_j)^2
            # Let dx_k = cx_i_k - cx_j_k, dy_k = cy_i_k - cy_j_k
            # Constraint: 2 * dx_k * (cx_i - cx_j) + 2 * dy_k * (cy_i - cy_j) - (r_i + r_j)^2 >= 0
            
            cx_k = x_current[:N]
            cy_k = x_current[N:2*N]
            r_k = x_current[2*N:]
            
            for i in range(N):
                for j in range(i+1, N):
                    dx_k = cx_k[i] - cx_k[j]
                    dy_k = cy_k[i] - cy_k[j]
                    # If distance is 0, this linearization is degenerate. 
                    # Add a small epsilon to avoid division by zero or undefined direction?
                    # If dx_k=0 and dy_k=0, the constraint is -(r_i+r_j)^2 >= 0, which is impossible if r>0.
                    # This forces the solver to move points apart.
                    
                    def make_overlap_constraint(i, j, dx_k, dy_k):
                        def constraint_func(x):
                            cx_i = x[i]
                            cx_j = x[j]
                            cy_i = x[N+i]
                            cy_j = x[N+j]
                            r_i = x[2*N+i]
                            r_j = x[2*N+j]
                            
                            lhs = 2 * dx_k * (cx_i - cx_j) + 2 * dy_k * (cy_i - cy_j)
                            rhs = (r_i + r_j) ** 2
                            
                            return lhs - rhs
                        return constraint_func
                    
                    constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j, dx_k, dy_k)})
            
            # Bounds
            bounds = []
            for i in range(N):
                bounds.append((0, W))
            for i in range(N):
                bounds.append((0, H))
            for i in range(N):
                bounds.append((0, min(W, H)/2))
            
            # Solve
            try:
                res = minimize(
                    objective,
                    x_current,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-9}
                )
                
                if res.success:
                    x_new = res.x
                else:
                    # If SLSQP fails, try to use the best point it found?
                    # Or break?
                    # Let's try to continue with x_current if it fails, but reduce lambda?
                    # For now, if it fails significantly, break.
                    if 'message' in res and 'Iteration limit reached' not in res.message:
                         # Try to use x_new if it's valid?
                         pass
                    x_new = res.x
                    
            except Exception:
                break
                
            # Check validity of x_new
            cx_new = x_new[:N]
            cy_new = x_new[N:2*N]
            r_new = x_new[2*N:]
            
            valid = True
            if np.any(cx_new - r_new < -1e-6) or np.any(cx_new + r_new > W + 1e-6):
                valid = False
            if np.any(cy_new - r_new < -1e-6) or np.any(cy_new + r_new > H + 1e-6):
                valid = False
            if np.any(r_new < -1e-6):
                valid = False
                
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist = math.hypot(cx_new[i] - cx_new[j], cy_new[i] - cy_new[j])
                        if dist < r_new[i] + r_new[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = np.sum(r_new)
                if score > best_score:
                    best_score = score
                    best_result = x_new.copy()
                
                # Check convergence
                diff = np.linalg.norm(x_new - x_current)
                if diff < 1e-6:
                    converged = True
                    break
                    
                x_current = x_new
            else:
                # If invalid, shrink lambda to prioritize feasibility (regularization)
                lambda_val *= 0.5
                if lambda_val < 1e-4:
                    break
                # Keep x_current as is for next iteration's linearization?
                # Or use x_new clipped?
                # Let's stick with x_current to ensure linearization is based on a valid point
                # Actually, if x_new is invalid, the linearization might be bad.
                # But SLSQP tried to satisfy linearized constraints.
                # Let's just continue with x_current.
                pass
                
        if converged:
            break
            
    # Fallback
    if best_result is None:
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.4 / side
                cells.append((cx, cy, r))
        best_result = np.array(cells[:N])
        
    return best_result

# EVOLVE-BLOCK-END
