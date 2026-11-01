"""Circle packing in perimeter-4 rectangle: Cellular Init + SLSQP + Jiggle Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _cellular_init(w, h, rows, cols, rng):
    """
    Initialize circles in a grid of cells.
    Returns x, y, r arrays.
    """
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # We have N circles. If rows*cols < N, we have extra circles.
    # We distribute them as best we can, or just take the first N cells if rows*cols >= N.
    # To be safe, we generate a grid that covers N cells.
    total_cells = rows * cols
    if total_cells < N:
        # Adjust grid to be at least N cells
        # Simple adjustment: increase cols or rows
        while rows * cols < N:
            cols += 1
        total_cells = rows * cols
        
    # Assign circles to the first N cells
    for i in range(N):
        row = i // cols
        col = i % cols
        
        # Cell boundaries
        x_min = col * (w / cols)
        x_max = (col + 1) * (w / cols)
        y_min = row * (h / rows)
        y_max = (row + 1) * (h / rows)
        
        # Center of cell
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        
        # Max radius is half the width or height of the cell
        cell_w = w / cols
        cell_h = h / rows
        max_r = min(cell_w, cell_h) / 2.0
        
        # Start with 90% of max radius to allow some freedom
        r0 = max_r * 0.9
        
        # Add small random jitter to break symmetry
        jitter_x = rng.uniform(-0.01, 0.01)
        jitter_y = rng.uniform(-0.01, 0.01)
        
        # Ensure circle stays inside cell initially
        cx = np.clip(cx + jitter_x, x_min + r0, x_max - r0)
        cy = np.clip(cy + jitter_y, y_min + r0, y_max - r0)
        
        x[i] = cx
        y[i] = cy
        r[i] = r0
        
    return x, y, r

def _force_relax(z, max_iter=1000, time_budget=30.0):
    """Force-directed relaxation with adaptive step and jitter."""
    z = z.copy().astype(float)
    n = N
    dt = 0.01
    min_dt = 1e-4
    rng = np.random.default_rng(123)
    
    for it in range(max_iter):
        if time.time() > time_budget:
            break
            
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w
        
        if w < 0.2 or h < 0.2:
            break
            
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50
        
        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        
        if np.max(overlap) > 1e-6:
            f_pairs = overlap * 20.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs
            
            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]
                
        # Check convergence
        if np.max(overlap) < 1e-7 and np.max(np.abs(left_viol)) < 1e-7:
            break
            
        # Update positions
        x += dt * fx
        y += dt * fy
        
        # Jitter to escape local minima (only if stuck or periodically)
        if it % 50 == 0 and it > 0:
            x += rng.normal(0, 0.001, n)
            y += rng.normal(0, 0.001, n)
            
        # Clamp positions
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        
        z[0::3][:n] = x
        z[1::3][:n] = y
        
        # Adaptive step size
        force_mag = np.max(np.sqrt(fx*fx + fy*fy))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        else:
            dt = min(0.05, dt * 1.1)
            
    return z

def _slsqp_refine(z, time_budget=20.0):
    """SLSQP optimization with explicit constraints."""
    z = z.copy().astype(float)
    n = N
    
    def objective(zv):
        r = zv[2::3][:n]
        return -np.sum(r)
        
    constraints = []
    bounds = []
    
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))
    
    # Boundary constraints
    for i in range(n):
        # Left: x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        # Right: w - (x + r) >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3*i] - p[3*i+2]})
        # Bottom: y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        # Top: h - (y + r) >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3*i+1] - p[3*i+2]})
        
    # Overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2
            })
            
    try:
        res = minimize(objective, z, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 500, 'ftol': 1e-10, 'maxls': 100})
        if res.success and _is_valid(res.x[0::3][:n], res.x[1::3][:n], res.x[2::3][:n], res.x[-1], 2.0-res.x[-1]):
            return res.x
    except Exception:
        pass
    return z

def _expand_radii(x, y, r, w, h, max_iter=500):
    """Iteratively expand radii as much as possible."""
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            # Calculate max possible radius for i
            r_max = min(x[i], w - x[i], y[i], h - y[i])
            for j in range(N):
                if i == j: continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(r_max, 0.0)
            
            if r_max > r[i] + 1e-8:
                r[i] = r_max
                improved = True
        if not improved:
            break
    return r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_value = 0.0
    best_result = None
    
    rng = np.random.default_rng(42)
    
    # Define different grid layouts to try
    layouts = [
        (5, 5),   # 25 cells, use 21
        (4, 6),   # 24 cells
        (6, 4),   # 24 cells
        (3, 7),   # 21 cells
        (7, 3),   # 21 cells
        (5, 4),   # 20 cells -> will expand
        (4, 5),   # 20 cells -> will expand
        (1, 21),  # 1 row
        (21, 1),  # 1 col
        (2, 11),  # 22 cells
        (11, 2),  # 22 cells
    ]
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.4, 0.7, 1.6, 0.65]
    
    attempts = 0
    for w0 in aspect_ratios:
        if time.time() > deadline - 30:
            break
            
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
            
        for rows, cols in layouts:
            if time.time() > deadline - 20:
                break
                
            attempts += 1
            try:
                x, y, r = _cellular_init(w0, h0, rows, cols, rng)
                
                # Construct z vector
                z = np.zeros(3*N + 1)
                z[0::3][:N] = x
                z[1::3][:N] = y
                z[2::3][:N] = r
                z[-1] = w0
                
                # Phase 1: Force Relax
                z_relaxed = _force_relax(z, max_iter=500, time_budget=time.time() + 5)
                
                # Phase 2: Expand Radii
                x_r = z_relaxed[0::3][:N]
                y_r = z_relaxed[1::3][:N]
                r_r = z_relaxed[2::3][:N]
                w_r = z_relaxed[-1]
                h_r = 2.0 - w_r
                
                r_expanded = _expand_radii(x_r, y_r, r_r, w_r, h_r, max_iter=200)
                
                z_expanded = z_relaxed.copy()
                z_expanded[2::3][:N] = r_expanded
                
                # Phase 3: SLSQP Refine
                z_opt = _slsqp_refine(z_expanded, time_budget=time.time() + 10)
                
                # Check validity and update best
                x_final = z_opt[0::3][:N]
                y_final = z_opt[1::3][:N]
                r_final = z_opt[2::3][:N]
                w_final = z_opt[-1]
                h_final = 2.0 - w_final
                
                if _is_valid(x_final, y_final, r_final, w_final, h_final):
                    val = np.sum(r_final)
                    if val > best_value:
                        best_value = val
                        best_result = z_opt.copy()
                        
            except Exception as e:
                continue

    # If no result found, fallback to a simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side * 0.9
        best_result = np.zeros(3*N + 1)
        best_result[0::3][:N] = x
        best_result[1::3][:N] = y
        best_result[2::3][:N] = r
        best_result[-1] = w

    return np.column_stack((best_result[0::3][:N], best_result[1::3][:N], best_result[2::3][:N]))
