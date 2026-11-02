"""Circle packing in perimeter-4 rectangle using L-BFGS-B optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def elapsed():
        return time.time() - start_time

    def check_deadline():
        if elapsed() > deadline:
            raise TimeoutError()

    def compute_score(w, circles):
        """Compute the sum of radii, return 0 if invalid."""
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return 0.0
        total_r = 0.0
        n = len(circles)
        for i in range(n):
            xi, yi, ri = circles[i]
            if ri <= 0:
                return 0.0
            if xi - ri < -1e-9 or xi + ri > w + 1e-9:
                return 0.0
            if yi - ri < -1e-9 or yi + ri > h + 1e-9:
                return 0.0
            total_r += ri
        # Check no overlap
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < ri + rj - 1e-9:
                    return 0.0
        return total_r

    def objective_and_grad(x):
        """
        Objective function and gradient for L-BFGS-B.
        x = [w, x1, y1, r1, ..., xN, yN, rN]
        """
        check_deadline()
        
        w = x[0]
        h = 2.0 - w
        circles = x[1:].reshape(N, 3)
        
        # Penalty weight for smooth constraints
        # We use a soft penalty to allow gradient flow
        alpha = 50.0 
        
        obj = 0.0
        grad = np.zeros_like(x)
        
        # 1. Boundary penalties
        # We want to maximize sum(r_i) - penalty
        # L-BFGS-B minimizes, so we minimize: -sum(r_i) + penalty
        
        for i in range(N):
            xi, yi, ri = circles[i]
            idx = 1 + i*3
            
            # Maximize r_i -> minimize -r_i
            grad[idx+2] -= 1.0
            obj -= ri
            
            # Boundary violations
            # Left: xi - ri < 0
            viol = xi - ri
            if viol < 0:
                p = alpha * viol**2
                obj += p
                g = 2 * alpha * viol
                grad[idx] += g   # d/dxi
                grad[idx+2] -= g # d/dri
                
            # Right: xi + ri > w
            viol = xi + ri - w
            if viol > 0:
                p = alpha * viol**2
                obj += p
                g = 2 * alpha * viol
                grad[idx] += g   # d/dxi
                grad[idx+2] += g # d/dri
                grad[0] -= g     # d/dw (w is in denominator of h? No, w is explicit)
                # Note: h = 2-w. Changing w affects h.
                # But boundary is xi+ri > w.
                
            # Bottom: yi - ri < 0
            viol = yi - ri
            if viol < 0:
                p = alpha * viol**2
                obj += p
                g = 2 * alpha * viol
                grad[idx+1] += g # d/dyi
                grad[idx+2] -= g # d/dri
                
            # Top: yi + ri > h
            viol = yi + ri - h
            if viol > 0:
                p = alpha * viol**2
                obj += p
                g = 2 * alpha * viol
                grad[idx+1] += g # d/dyi
                grad[idx+2] += g # d/dri
                # d/dw: h = 2-w, so d/dw (violation) = d/dw (yi + ri - (2-w)) = +1
                grad[0] += g     # d/dw
        
        # 2. Overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                idx_i = 1 + i*3
                idx_j = 1 + j*3
                
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                sum_r = ri + rj
                
                viol = sum_r - dist
                if viol > 0:
                    p = alpha * viol**2
                    obj += p
                    g = 2 * alpha * viol
                    
                    # Gradients of dist
                    # d(dist)/dx_i = dx/dist
                    # d(dist)/dx_j = -dx/dist
                    # d(dist)/dy_i = dy/dist
                    # d(dist)/dy_j = -dy/dist
                    # d(dist)/dr_i = 0 (dist independent of r)
                    
                    # Gradient of violation:
                    # d(viol)/dx_i = -dx/dist
                    # d(viol)/dx_j = +dx/dist
                    # d(viol)/dy_i = -dy/dist
                    # d(viol)/dy_j = +dy/dist
                    # d(viol)/dr_i = 1
                    # d(viol)/dr_j = 1
                    
                    grad[idx_i] += g * (-dx/dist)
                    grad[idx_i+1] += g * (-dy/dist)
                    grad[idx_i+2] += g * 1.0
                    
                    grad[idx_j] += g * (dx/dist)
                    grad[idx_j+1] += g * (dy/dist)
                    grad[idx_j+2] += g * 1.0
                    
        return obj, grad

    def init_hexagon(w, h):
        """Initialize circles in a hexagonal lattice."""
        circles = []
        # Estimate radius from hex packing density
        # Area = N * pi * r^2. Hex density ~ 0.9069
        # Approx r ~ sqrt(Area / (N * pi / 0.9069))
        # But we just need a valid start.
        
        # Try to fit as many rows as possible
        # Hex spacing: dx = 2r, dy = sqrt(3)r
        # Let's pick a r that fits roughly
        # For N=21, a 5x5 grid with some removed or 6x4 etc.
        # Let's just use a grid that fits and set r to 0.49 * min(w/cols, h/rows)
        # But hex is better.
        
        # Simple approach: Grid init is safer for validity, but hex is better for score.
        # Let's try a few configurations.
        best_circles = None
        best_score = -1
        
        for rows in range(3, 8):
            # Hex rows: alternate offset
            # cols per row
            # To keep it simple and robust, let's stick to a rectangular grid
            # but with hexagonal spacing where possible.
            # Actually, simple grid init is sufficient for L-BFGS to find local minima.
            # The "inspiration" said hex is better. Let's implement a basic hex init.
            
            # Calculate max r for a hex lattice of 'rows'
            # Height = rows * sqrt(3) * r
            # Width = (cols + 0.5) * 2 * r  (approx)
            
            # Let's just use a standard grid for init to ensure validity,
            # then let L-BFGS optimize. The optimization will move them to hex.
            cols = int(np.ceil(N / rows))
            if rows * cols < N:
                continue
            
            # Calculate r
            r = 0.45 * min(w / cols, h / rows)
            
            circles = []
            for i in range(rows):
                for j in range(cols):
                    if len(circles) >= N:
                        break
                    cx = (j + 0.5) * w / cols
                    cy = (i + 0.5) * h / rows
                    circles.append([cx, cy, r])
            
            # Check validity
            if len(circles) == N:
                score = compute_score(w, circles)
                if score > best_score:
                    best_score = score
                    best_circles = circles
        
        if best_circles is None:
             # Fallback
             side = int(np.ceil(np.sqrt(N)))
             best_circles = []
             for i in range(side):
                 for j in range(side):
                     if len(best_circles) >= N:
                         break
                     cx = (j + 0.5) / side
                     cy = (i + 0.5) / side
                     r = 0.45 / side
                     best_circles.append([cx, cy, r])
        
        return best_circles

    best_score = 0.0
    best_result = None

    # Try several aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0]
    
    for ar in aspect_ratios:
        if elapsed() > deadline - 20:
            break
        
        w = ar / (1 + ar) * 2.0
        h = 2.0 - w
        
        circles = init_hexagon(w, h)
        if circles is None or len(circles) != N:
            continue
            
        x0 = np.array([w] + [v for c in circles for v in c])
        
        # Bounds
        # w in [0.1, 1.9]
        # xi in [0, w] (loose bounds for optimizer, penalty handles it)
        # yi in [0, h]
        # ri in [0.01, 1.0]
        
        bounds = [(0.1, 1.9)]
        for i in range(N):
            bounds.append((0.0, w)) # xi
            bounds.append((0.0, h)) # yi
            bounds.append((0.01, 1.0)) # ri
            
        # Optimize
        try:
            res = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
            )
        except Exception:
            continue

        x_opt = res.x
        w_opt = x_opt[0]
        circles_opt = x_opt[1:].reshape(N, 3)
        
        # Check validity of the result
        # L-BFGS-B might end up with slight violations due to numerical errors
        # We need to clamp or repair.
        
        # Simple repair: clamp centers and radii
        repaired_circles = []
        valid = True
        for i in range(N):
            xi, yi, ri = circles_opt[i]
            # Clamp r
            ri = max(0.01, min(ri, 1.0))
            # Clamp x, y to keep circle inside
            # This might cause overlap, but it's a start
            xi = max(ri, min(xi, w_opt - ri))
            yi = max(ri, min(yi, (2.0 - w_opt) - ri))
            repaired_circles.append([xi, yi, ri])
            
        score = compute_score(w_opt, repaired_circles)
        
        if score > best_score:
            best_score = score
            best_result = (w_opt, repaired_circles)

    # If we have time, do a second pass with refined aspect ratios around the best one
    if best_result is not None and elapsed() < deadline - 30:
        w_best, circles_best = best_result
        # Try small perturbations of w
        for delta_w in [-0.1, -0.05, 0.05, 0.1]:
            w_try = w_best + delta_w
            if w_try < 0.1 or w_try > 1.9:
                continue
            h_try = 2.0 - w_try
            
            x0 = np.array([w_try] + [v for c in circles_best for v in c])
            bounds = [(0.1, 1.9)]
            for i in range(N):
                bounds.append((0.0, w_try))
                bounds.append((0.0, h_try))
                bounds.append((0.01, 1.0))
                
            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                x_opt = res.x
                w_opt = x_opt[0]
                circles_opt = x_opt[1:].reshape(N, 3)
                
                repaired_circles = []
                for i in range(N):
                    xi, yi, ri = circles_opt[i]
                    ri = max(0.01, min(ri, 1.0))
                    xi = max(ri, min(xi, w_opt - ri))
                    yi = max(ri, min(yi, (2.0 - w_opt) - ri))
                    repaired_circles.append([xi, yi, ri])
                    
                score = compute_score(w_opt, repaired_circles)
                if score > best_score:
                    best_score = score
                    best_result = (w_opt, repaired_circles)
            except Exception:
                pass

    # Build output
    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    w, circles = best_result
    result = np.array([[c[0], c[1], c[2]] for c in circles])
    return result

# EVOLVE-BLOCK-END
