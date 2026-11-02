import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.spatial import Delaunay, QhullError
import time

N = 21

def _softplus(z, k=10.0):
    """Smooth approximation of max(0, z)."""
    return np.logaddexp(0.0, k * z) / k

def _solve_lp_radii(xs, ys, w, h, max_iter=200):
    """
    Given fixed centers and rectangle, solve for optimal radii using LP.
    Maximize sum(r_i) subject to:
      r_i + r_j <= dist_ij  for all pairs (i,j)
      r_i <= xs[i], r_i <= w - xs[i], r_i <= ys[i], r_i <= h - ys[i]
      r_i >= 0
    """
    n = len(xs)
    
    # Build constraint matrix: A @ r <= b
    # We want to maximize sum(r), so minimize -sum(r)
    # Constraints: r_i + r_j <= d_ij, r_i <= bound_ij
    
    constraints = []
    rhs = []
    
    # Pair constraints
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            constraints.append(row)
            rhs.append(d)
    
    # Boundary constraints
    for i in range(n):
        bounds_i = [xs[i], w - xs[i], ys[i], h - ys[i]]
        for b in bounds_i:
            row = np.zeros(n)
            row[i] = 1.0
            constraints.append(row)
            rhs.append(max(b, 0.0))
    
    if not constraints:
        return np.zeros(n)
    
    A = np.array(constraints)
    b = np.array(rhs)
    
    # Use scipy linprog
    from scipy.optimize import linprog
    
    c = -np.ones(n)  # minimize -sum(r) = maximize sum(r)
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * n, method='highs')
    
    if res.success:
        return np.maximum(res.x, 0.0)
    else:
        # Fallback: just use minimum of distances to boundaries
        rs = np.zeros(n)
        for i in range(n):
            rs[i] = min(xs[i], w - xs[i], ys[i], h - ys[i])
            # Also check pair distances
            for j in range(n):
                if i != j:
                    d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                    rs[i] = min(rs[i], d / 2.0)
        return np.maximum(rs, 0.0)

def _check_validity(xs, ys, rs, w, h, tol=1e-7):
    """Check if the packing is valid."""
    if 2 * (w + h) > 4.0 + tol:
        return False
    for i in range(len(xs)):
        if xs[i] < -tol or xs[i] > w + tol:
            return False
        if ys[i] < -tol or ys[i] > h + tol:
            return False
        if rs[i] < -tol:
            return False
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            if d < rs[i] + rs[j] - tol:
                return False
    return True

def _make_hex(w, h):
    """Create a hexagonal packing initialization."""
    # Try to fit a hexagonal pattern
    # For a hexagonal lattice, circles of radius r have centers at:
    # (i * 2r, j * sqrt(3)*r + (i%2)*sqrt(3)*r/2)
    
    circles = []
    # Start with a reasonable radius estimate
    # Area of rectangle / N gives average area per circle
    avg_area = w * h / N
    r_est = np.sqrt(avg_area / np.pi)
    
    # Try different radii
    for r in [r_est * 0.8, r_est * 0.9, r_est, r_est * 1.1]:
        # Place circles in hexagonal pattern
        dx = 2 * r
        dy = np.sqrt(3) * r
        
        for row in range(int(h / dy) + 2):
            for col in range(int(w / dx) + 2):
                x = col * dx + (row % 2) * dx / 2
                y = row * dy
                if x + r <= w and y + r <= h and x - r >= 0 and y - r >= 0:
                    circles.append((x, y, r))
                if len(circles) >= N:
                    break
            if len(circles) >= N:
                break
        
        if len(circles) >= N:
            circles = circles[:N]
            xs = np.array([c[0] for c in circles])
            ys = np.array([c[1] for c in circles])
            rs = np.array([c[2] for c in circles])
            # Check if valid, if so return
            if _check_validity(xs, ys, rs, w, h):
                return xs, ys, rs
    
    # If hexagonal doesn't work, return empty
    return None

def _make_grid(w, h):
    """Create a grid packing initialization."""
    # Try different grid configurations
    best = None
    best_val = -np.inf
    
    for rows in range(3, 8):
        for cols in range(3, 8):
            if rows * cols < N:
                continue
            
            cell_w = w / cols
            cell_h = h / rows
            
            # Place circles in grid
            circles = []
            for i in range(rows):
                for j in range(cols):
                    x = (j + 0.5) * cell_w
                    y = (i + 0.5) * cell_h
                    r = min(cell_w, cell_h) / 2 - 1e-4
                    circles.append((x, y, r))
            
            if len(circles) >= N:
                circles = circles[:N]
                xs = np.array([c[0] for c in circles])
                ys = np.array([c[1] for c in circles])
                rs = np.array([c[2] for c in circles])
                
                if _check_validity(xs, ys, rs, w, h):
                    val = np.sum(rs)
                    if val > best_val:
                        best_val = val
                        best = (xs, ys, rs)
    
    return best

def _make_random(w, h, n_attempts=100):
    """Create random packing initializations."""
    best = None
    best_val = -np.inf
    
    for _ in range(n_attempts):
        xs = np.random.uniform(0.1, w - 0.1, N)
        ys = np.random.uniform(0.1, h - 0.1, N)
        
        # Solve for optimal radii
        rs = _solve_lp_radii(xs, ys, w, h)
        
        if _check_validity(xs, ys, rs, w, h):
            val = np.sum(rs)
            if val > best_val:
                best_val = val
                best = (xs, ys, rs)
    
    return best

def _refine_with_slsqp(xs, ys, rs, w, h, time_limit=30):
    """Refine the packing using SLSQP."""
    start_time = time.time()
    
    def objective(x):
        w_x, h_x = x[0], x[1]
        xs_x = x[2:2 + N]
        ys_x = x[2 + N:2 + 2 * N]
        rs_x = x[2 + 2 * N:2 + 3 * N]
        
        # Penalty for perimeter violation
        perim_viol = max(0, 2 * (w_x + h_x) - 4.0)
        
        # Penalty for overlaps
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((xs_x[i] - xs_x[j])**2 + (ys_x[i] - ys_x[j])**2)
                viol = rs_x[i] + rs_x[j] - d
                if viol > 0:
                    overlap_pen += viol**2
        
        # Penalty for boundary violations
        bound_pen = 0.0
        for i in range(N):
            bound_pen += max(0, rs_x[i] - xs_x[i])**2
            bound_pen += max(0, xs_x[i] + rs_x[i] - w_x)**2
            bound_pen += max(0, rs_x[i] - ys_x[i])**2
            bound_pen += max(0, ys_x[i] + rs_x[i] - h_x)**2
        
        return -np.sum(rs_x) + 100 * (perim_viol**2 + overlap_pen + bound_pen)
    
    # Initial point
    x0 = np.concatenate(([w, h], xs, ys, rs))
    
    # Bounds
    bounds = [(0.1, 1.9), (0.1, 1.9)]
    for _ in range(N):
        bounds.append((0, 2))
    for _ in range(N):
        bounds.append((0, 2))
    for _ in range(N):
        bounds.append((1e-4, 1.0))
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                      options={'maxiter': 200, 'ftol': 1e-10})
        
        if res.success:
            w_r, h_r = res.x[0], res.x[1]
            xs_r = res.x[2:2 + N]
            ys_r = res.x[2 + N:2 + 2 * N]
            rs_r = res.x[2 + 2 * N:2 + 3 * N]
            
            if _check_validity(xs_r, ys_r, rs_r, w_r, h_r):
                return xs_r, ys_r, rs_r, w_r, h_r
    except Exception:
        pass
    
    return None

def _scale_to_perimeter(xs, ys, rs, w, h):
    """Scale the packing to have perimeter exactly 4."""
    current_perim = 2 * (w + h)
    if current_perim <= 0:
        return xs, ys, rs, w, h
    
    scale = 4.0 / current_perim
    w_new = w * scale
    h_new = h * scale
    xs_new = xs * scale
    ys_new = ys * scale
    rs_new = rs * scale
    
    return xs_new, ys_new, rs_new, w_new, h_new

def construct():
    start_time = time.time()
    time_limit = 140  # Safety margin for 150s kill
    
    best_val = -np.inf
    best_result = None
    
    # Try different rectangle aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6]
    
    for ar in aspect_ratios:
        if time.time() - start_time > time_limit:
            break
        
        # w + h = 2 (since perimeter = 4)
        # w / h = ar => w = ar * h => ar*h + h = 2 => h = 2/(ar+1), w = 2*ar/(ar+1)
        h = 2.0 / (ar + 1.0)
        w = 2.0 * ar / (ar + 1.0)
        
        # Try different initialization strategies
        inits = []
        
        # Hexagonal
        hex_init = _make_hex(w, h)
        if hex_init is not None:
            inits.append(('hex', hex_init))
        
        # Grid
        grid_init = _make_grid(w, h)
        if grid_init is not None:
            inits.append(('grid', grid_init))
        
        # Random
        rand_init = _make_random(w, h, n_attempts=50)
        if rand_init is not None:
            inits.append(('random', rand_init))
        
        # Process each initialization
        for init_name, (xs, ys, rs) in inits:
            if time.time() - start_time > time_limit:
                break
            
            # Solve for optimal radii with fixed centers
            rs_opt = _solve_lp_radii(xs, ys, w, h)
            
            if _check_validity(xs, ys, rs_opt, w, h):
                val = np.sum(rs_opt)
                if val > best_val:
                    best_val = val
                    best_result = np.column_stack((xs, ys, rs_opt))
            
            # Try SLSQP refinement if time allows
            if time.time() - start_time < time_limit - 20:
                refined = _refine_with_slsqp(xs, ys, rs_opt, w, h, time_limit=15)
                if refined is not None:
                    xs_r, ys_r, rs_r, w_r, h_r = refined
                    if _check_validity(xs_r, ys_r, rs_r, w_r, h_r):
                        val = np.sum(rs_r)
                        if val > best_val:
                            best_val = val
                            best_result = np.column_stack((xs_r, ys_r, rs_r))
    
    # Final polish: try to improve with more iterations
    if best_result is not None and time.time() - start_time < time_limit - 10:
        xs = best_result[:, 0]
        ys = best_result[:, 1]
        rs = best_result[:, 2]
        
        # Estimate w and h from the bounding box
        w_est = max(xs + rs)
        h_est = max(ys + rs)
        
        # Scale to perimeter 4
        xs, ys, rs, w, h = _scale_to_perimeter(xs, ys, rs, w_est, h_est)
        
        # Try SLSQP refinement
        refined = _refine_with_slsqp(xs, ys, rs, w, h, time_limit=10)
        if refined is not None:
            xs_r, ys_r, rs_r, w_r, h_r = refined
            if _check_validity(xs_r, ys_r, rs_r, w_r, h_r):
                val = np.sum(rs_r)
                if val > best_val:
                    best_val = val
                    best_result = np.column_stack((xs_r, ys_r, rs_r))
    
    if best_result is not None:
        return best_result
    else:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                for i in range(side) for j in range(side)]
        return np.array(cells[:N])
