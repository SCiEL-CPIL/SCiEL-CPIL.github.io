"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START

_deadline = 0.0

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    circles = np.array(circles)
    if len(circles) != N:
        return False
    for i in range(N):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    # Vectorized overlap check
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    for i in range(N):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(dist < r[i] + r[i+1:] - tol):
            return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _make_hexagonal_grid(n, w, h, row_counts):
    """
    Create an initial hexagonal-like arrangement.
    row_counts: list of integers summing to n, representing circles per row.
    """
    circles = []
    total_rows = len(row_counts)
    
    # Estimate radius based on area packing fraction (hexagonal is ~0.9069)
    # Area of rect = w*h. Sum of areas of circles <= w*h * 0.9
    # n * pi * r^2 <= w*h * 0.9  => r <= sqrt(w*h*0.9 / (n*pi))
    r_est = np.sqrt(w * h * 0.9 / (n * np.pi)) * 0.95 # 0.95 safety factor
    
    # Vertical spacing
    # In hexagonal packing, vertical distance between rows is r * sqrt(3)
    # But here we have variable counts per row, so we just distribute y evenly
    # and rely on relaxation to fix overlaps.
    for row_i, count in enumerate(row_counts):
        y = (row_i + 0.5) * h / total_rows
        # Horizontal spacing
        # If count > 0, distribute across width
        # Offset alternating rows for hexagonal effect
        offset = 0.0
        if total_rows > 1 and row_i % 2 == 1:
            # Simple offset: half of the average spacing of the row above/below
            # For simplicity, just shift by half a cell width if we assume regularity
            # A better heuristic: if the row above had count 'c_above', spacing is w/c_above.
            # We don't have easy access to 'c_above' in this simple loop without pre-calc.
            # Let's just use a fixed fraction of w/n as offset
            offset = 0.5 * w / n
            
        for col_j in range(count):
            # Center the row horizontally
            # x positions: (j + 0.5) / count * w + offset
            # But we need to ensure they fit. 
            # If count is small, spacing is large.
            # Let's just place them evenly in [offset, w-offset]
            if count == 1:
                x = w / 2.0
            else:
                x = offset + (col_j + 0.5) * (w - 2*offset) / count
            
            # Ensure x is within bounds
            x = np.clip(x, r_est, w - r_est)
            
            # Radius: start slightly smaller to ensure initial validity
            r = r_est * 0.8
            # Check boundary constraints for this specific position
            r = min(r, x, w - x, y, h - y)
            r = max(r, 0.001)
            
            circles.append([x, y, r])
            
    # If we have more or fewer than n, adjust
    if len(circles) < n:
        # Add tiny circles in center? Or just pad
        while len(circles) < n:
            circles.append([w/2, h/2, 0.001])
    else:
        circles = circles[:n]
        
    return np.array(circles)

def _penalty_optimize(circles, w, h, deadline, beta_start=50.0, beta_end=5000.0, n_beta_steps=10):
    """
    Penalty-based optimization using L-BFGS-B.
    Structural change: Uses smooth penalty functions for constraints instead of hard SLSQP constraints.
    """
    if time.time() > deadline - 5.0:
        return None

    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    # Bounds for L-BFGS-B
    # x, y are bounded by [0, w], [0, h]
    # r is bounded by [0, min(w,h)/2]
    bounds = []
    for i in range(n):
        bounds.append((0, w))
        bounds.append((0, h))
    for i in range(n):
        bounds.append((1e-5, min(w, h)/2))

    def objective_and_grad(x, beta, eps):
        cx = x[:n]
        cy = x[n:2*n]
        r = x[2*n:]
        
        obj = -np.sum(r)
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0
        
        # Boundary penalty (smooth log-barrier style or quadratic penalty for violations)
        # Using a smooth penalty: if val < 0, add beta * val^2 / (2*eps) ? 
        # Or simply: penalty = beta * max(0, -val)^2 is not differentiable at 0.
        # Smooth approximation: p(z) = 0.5 * beta * (z + sqrt(z^2 + eps^2)) - 0.5 * beta * eps
        # where z is the violation amount (positive if violation).
        # Here, constraints are: cx - r >= 0 => violation = r - cx
        # Let's use a simpler quadratic penalty with smoothing for gradients.
        # If g(x) >= 0 is the constraint, penalty is beta * max(0, -g(x))^2.
        # To make it smooth, we can use: p(g) = 0.5 * beta * (max(0, -g) + sqrt(max(0, -g)^2 + eps^2))^2 - ... 
        # Actually, standard smooth penalty: 
        # P(t) = 0.5 * t^2 for t >= 0
        # P(t) = 0.5 * beta * (t + sqrt(t^2 + eps^2))^2 ... no.
        
        # Let's use the "soft" penalty: 
        # If constraint is h(x) >= 0.
        # Penalty term: 0.5 * beta * (max(0, -h(x)))^2 is C1 but not C2.
        # For L-BFGS-B, C1 is often sufficient.
        # Let's just use 0.5 * beta * (max(0, -h))^2.
        
        # Boundary constraints:
        # 1. cx - r >= 0
        # 2. w - cx - r >= 0
        # 3. cy - r >= 0
        # 4. h - cy - r >= 0
        
        for i in range(n):
            # Left
            val = cx[i] - r[i]
            if val < 0:
                pen = 0.5 * beta * val * val
                dpen_dval = beta * val
                obj += pen
                grad[i] += dpen_dval
                grad[2*n+i] -= dpen_dval
            
            # Right
            val = w - cx[i] - r[i]
            if val < 0:
                pen = 0.5 * beta * val * val
                dpen_dval = beta * val
                obj += pen
                grad[i] -= dpen_dval
                grad[2*n+i] -= dpen_dval
            
            # Bottom
            val = cy[i] - r[i]
            if val < 0:
                pen = 0.5 * beta * val * val
                dpen_dval = beta * val
                obj += pen
                grad[n+i] += dpen_dval
                grad[2*n+i] -= dpen_dval
            
            # Top
            val = h - cy[i] - r[i]
            if val < 0:
                pen = 0.5 * beta * val * val
                dpen_dval = beta * val
                obj += pen
                grad[n+i] -= dpen_dval
                grad[2*n+i] -= dpen_dval

        # Overlap constraints: dist(i,j) - r[i] - r[j] >= 0
        # Vectorized
        cx_exp = cx[:, np.newaxis]
        cy_exp = cy[:, np.newaxis]
        r_exp = r[:, np.newaxis]
        
        dx = cx_exp - cx_exp.T
        dy = cy_exp - cy_exp.T
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        
        # We only need i < j
        # Create mask
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        
        # Get values
        dist_vals = dist[mask]
        r_i = r[np.triu_indices(n, k=1)[0]]
        r_j = r[np.triu_indices(n, k=1)[1]]
        
        # Constraint: dist - r_i - r_j >= 0
        gap = dist_vals - r_i - r_j
        
        # Penalty for gap < 0
        # Only penalize if gap < 0
        neg_gap = np.maximum(0, -gap)
        pen_vals = 0.5 * beta * neg_gap * neg_gap
        obj += np.sum(pen_vals)
        
        # Gradient
        # d(pen)/d(gap) = beta * neg_gap * d(neg_gap)/d(gap)
        # If gap < 0, neg_gap = -gap, d(neg_gap)/d(gap) = -1.
        # So d(pen)/d(gap) = -beta * (-gap) = beta * gap ? No.
        # pen = 0.5 * beta * (-gap)^2. d(pen)/d(gap) = beta * (-gap) * (-1) = beta * gap.
        # Wait. If gap < 0, -gap > 0.
        # Let u = -gap. pen = 0.5 * beta * u^2. d(pen)/du = beta * u = beta * (-gap).
        # d(gap)/dx ...
        
        # Let's compute gradients for each pair
        # This loop is O(N^2) but N=21, so ~200 iterations. Fast enough.
        # However, vectorizing the gradient update is tricky with the mask.
        # Let's iterate over the upper triangle indices.
        
        idx_i, idx_j = np.triu_indices(n, k=1)
        
        for k in range(len(idx_i)):
            i = idx_i[k]
            j = idx_j[k]
            
            if gap[k] < 0:
                # d(pen)/d(gap) = beta * (-gap[k]) * (-1) = beta * gap[k] ? 
                # No. pen = 0.5 * beta * (r_i + r_j - dist)^2.
                # Let v = r_i + r_j - dist. If v > 0, penalty is 0.5 * beta * v^2.
                # d(pen)/dv = beta * v.
                # dv/d(cx_i) = -1/dist * (cx_i - cx_j) ? No.
                # dist = sqrt((cx_i-cx_j)^2 + ...).
                # d(dist)/d(cx_i) = (cx_i - cx_j) / dist.
                # dv/d(cx_i) = - d(dist)/d(cx_i) = -(cx_i - cx_j) / dist.
                # d(pen)/d(cx_i) = beta * v * (-(cx_i - cx_j) / dist).
                
                v = r[i] + r[j] - dist[i, j]
                if v > 0:
                    dpen_dv = beta * v
                    ux = (cx[i] - cx[j]) / dist[i, j]
                    uy = (cy[i] - cy[j]) / dist[i, j]
                    
                    # grad(cx_i) += dpen_dv * (-ux)
                    # grad(cx_j) += dpen_dv * (ux)
                    grad[i] += dpen_dv * (-ux)
                    grad[j] += dpen_dv * (ux)
                    grad[n+i] += dpen_dv * (-uy)
                    grad[n+j] += dpen_dv * (uy)
                    
                    # grad(r_i) += dpen_dv * (1)
                    # grad(r_j) += dpen_dv * (1)
                    grad[2*n+i] += dpen_dv
                    grad[2*n+j] += dpen_dv

        return obj, grad

    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    x = x0.copy()
    
    for step in range(n_beta_steps):
        if time.time() > deadline - 2.0:
            break
        
        beta = betas[step]
        
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta, 1e-4)[0],
                x,
                jac=lambda x: objective_and_grad(x, beta, 1e-4)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            if res.success or res.fun < 0:
                x = res.x
        except Exception:
            break
    
    # Final check
    best_circles = np.array([
        [x[i], x[n+i], x[2*n+i]] for i in range(n)
    ])
    
    # Clamp to ensure strict validity for return
    for i in range(n):
        best_circles[i, 0] = np.clip(best_circles[i, 0], best_circles[i, 2], w - best_circles[i, 2])
        best_circles[i, 1] = np.clip(best_circles[i, 1], best_circles[i, 2], h - best_circles[i, 2])
        
    if _validate(best_circles, w, h, tol=1e-6):
        return best_circles
    return None

def _polish_slsqp(circles, w, h, deadline):
    """Final polish with SLSQP for higher precision and strict constraint satisfaction."""
    if time.time() > deadline - 3.0:
        return circles
    
    n = len(circles)
    x0 = np.array([v for c in circles for v in c])
    
    def objective(x):
        if time.time() > deadline:
            return 0.0
        r = x[2::3]
        return -np.sum(r)
    
    def objective_grad(x):
        if time.time() > deadline:
            return np.zeros_like(x)
        grad = np.zeros_like(x)
        grad[2::3] = -1.0
        return grad
    
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2]})
    
    for i in range(n):
        for j in range(i + 1, n):
            def overlap_constraint(x, i=i, j=j):
                dx = x[i*3] - x[j*3]
                dy = x[i*3+1] - x[j*3+1]
                return np.sqrt(dx*dx + dy*dy) - x[i*3+2] - x[j*3+2]
            constraints.append({'type': 'ineq', 'fun': overlap_constraint})
    
    bounds = [(0, w)] * n + [(0, h)] * n + [(0.001, 1.0)] * n
    
    try:
        res = minimize(
            objective,
            x0,
            jac=objective_grad,
            args=(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
        )
        
        if res.success or res.fun < 0:
            best_circles = [(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)]
            if _validate(best_circles, w, h, tol=1e-8):
                return np.array(best_circles)
    except Exception:
        pass
    
    return circles

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_circles = None
    best_score = 0.0
    
    # Aspect ratios: w + h = 2
    aspect_ratios = [
        (1.0, 1.0),
        (0.9, 1.1),
        (1.1, 0.9),
        (0.8, 1.2),
        (1.2, 0.8),
        (0.75, 1.25),
        (1.25, 0.75),
    ]
    
    # Row configurations for hexagonal grid (sum to 21)
    # 5-4-5-4-3, 4-5-4-5-3, 5-5-5-3-3, etc.
    grid_configs = [
        [5, 4, 5, 4, 3],
        [4, 5, 4, 5, 3],
        [5, 5, 5, 3, 3],
        [3, 5, 5, 5, 3],
        [6, 4, 5, 4, 2],
        [2, 4, 5, 4, 6],
        [7, 4, 4, 4, 2],
    ]
    
    for w, h in aspect_ratios:
        if time.time() > _deadline - 10.0:
            break
        
        for row_counts in grid_configs:
            if time.time() > _deadline - 5.0:
                break
            
            # Create initial grid
            circles = _make_hexagonal_grid(N, w, h, row_counts)
            
            # Penalty Optimization
            if time.time() < _deadline - 5.0:
                optimized = _penalty_optimize(circles, w, h, _deadline)
                if optimized is not None:
                    # Polish with SLSQP
                    polished = _polish_slsqp(optimized, w, h, _deadline)
                    score = _total_radius(polished)
                    if score > best_score:
                        best_score = score
                        best_circles = polished
    
    if best_circles is not None:
        return np.array(best_circles)
    
    # Fallback: 5x5 grid in unit square
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
