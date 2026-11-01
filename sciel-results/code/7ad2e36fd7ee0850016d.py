"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition with variable cells."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _objective_widths(z, NX, NY, active_mask):
    """
    Objective function for grid optimization using cell widths and heights.
    z layout: [u_0, ..., u_{NX-1}, v_0, ..., v_{NY-1}]
    """
    u = z[0:NX]
    v = z[NX:NX+NY]
    
    # Sum of radii for active cells
    # active_mask is a boolean array of shape (NY, NX)
    total_r = 0.0
    for r in range(NY):
        for c in range(NX):
            if active_mask[r, c]:
                total_r += min(u[c], v[r]) / 2.0
                
    return -total_r

def _constraint_sum(z, NX, NY):
    """
    Constraint: Sum of all widths and heights must be 2 (since 2(W+H)=4 => W+H=2).
    """
    return np.sum(z) - 2.0

def _optimize_grid_widths(NX, NY, active_mask, time_budget, w_init=1.0):
    """
    Optimize grid cell widths and heights for a given topology.
    """
    # Initial guess: uniform distribution of W=w_init and H=2-w_init
    w_init = np.clip(w_init, 0.5, 1.5)
    h_init = 2.0 - w_init
    
    u0 = np.full(NX, w_init / NX)
    v0 = np.full(NY, h_init / NY)
    z0 = np.concatenate([u0, v0])
    
    # Bounds
    bounds = [(1e-4, 1.5) for _ in range(NX + NY)]
    
    # Constraint
    constraints = [{'type': 'eq', 'fun': lambda z: _constraint_sum(z, NX, NY)}]
    
    def obj(z):
        return _objective_widths(z, NX, NY, active_mask)
        
    # Use SLSQP
    maxiter = int(min(500, time_budget * 30))
    
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 200})
        
        if res.success or res.fun < obj(z0):
            z_opt = res.x
        else:
            z_opt = z0
    except Exception:
        z_opt = z0
        
    u_opt = z_opt[0:NX]
    v_opt = z_opt[NX:NX+NY]
    
    # Calculate centers and radii
    centers = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Cumulative sums for positions
    x_starts = np.concatenate(([0.0], np.cumsum(u_opt)[:-1]))
    y_starts = np.concatenate(([0.0], np.cumsum(v_opt)[:-1]))
    
    idx = 0
    for r in range(NY):
        for c in range(NX):
            if active_mask[r, c]:
                cx = x_starts[c] + u_opt[c] / 2.0
                cy = y_starts[r] + v_opt[r] / 2.0
                r_cell = min(u_opt[c], v_opt[r]) / 2.0
                
                centers[idx] = [cx, cy]
                radii[idx] = r_cell
                idx += 1
                
    return centers, radii, u_opt, v_opt

def _check_validity(x, y, r, w, h, tol=1e-9):
    """Check if the packing is valid."""
    if w <= 0.05 or h <= 0.05:
        return False, 0.0
    for i in range(N):
        if r[i] <= 0:
            return False, 0.0
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False, 0.0
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False, 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False, 0.0
    return True, float(np.sum(r))

def construct():
    start_time = time.time()
    deadline = start_time + 105.0
    
    best_value = 0.0
    best_result = None
    
    # Define topologies: (NX, NY, active_mask, w_init)
    # active_mask is a list of (r, c) tuples that are ACTIVE
    
    topologies = []
    
    # 1. 5x5 grid, remove 4 corners (indices (0,0), (0,4), (4,0), (4,4))
    mask_5x5_corners = np.ones((5, 5), dtype=bool)
    mask_5x5_corners[0, 0] = False
    mask_5x5_corners[0, 4] = False
    mask_5x5_corners[4, 0] = False
    mask_5x5_corners[4, 4] = False
    topologies.append((5, 5, mask_5x5_corners, 1.0))
    topologies.append((5, 5, mask_5x5_corners, 0.8))
    topologies.append((5, 5, mask_5x5_corners, 1.2))
    
    # 2. 5x5 grid, remove 4 edges (e.g., middle of each side)
    mask_5x5_edges = np.ones((5, 5), dtype=bool)
    mask_5x5_edges[0, 2] = False
    mask_5x5_edges[4, 2] = False
    mask_5x5_edges[2, 0] = False
    mask_5x5_edges[2, 4] = False
    topologies.append((5, 5, mask_5x5_edges, 1.0))
    
    # 3. 7x3 grid, all active
    mask_7x3 = np.ones((3, 7), dtype=bool)
    topologies.append((7, 3, mask_7x3, 1.4)) # Wider
    topologies.append((7, 3, mask_7x3, 1.2))
    
    # 4. 3x7 grid, all active
    mask_3x7 = np.ones((7, 3), dtype=bool)
    topologies.append((3, 7, mask_3x7, 0.8)) # Taller
    topologies.append((3, 7, mask_3x7, 1.0))
    
    # 5. 6x4 grid, remove 3 corners
    mask_6x4 = np.ones((4, 6), dtype=bool)
    mask_6x4[0, 0] = False
    mask_6x4[0, 5] = False
    mask_6x4[3, 0] = False
    topologies.append((6, 4, mask_6x4, 1.2))
    
    for NX, NY, mask, w_init in topologies:
        if time.time() > deadline - 5:
            break
            
        remaining = deadline - (time.time() - start_time)
        if remaining < 2:
            break
            
        try:
            centers, radii, u_opt, v_opt = _optimize_grid_widths(NX, NY, mask, remaining, w_init)
            
            w = np.sum(u_opt)
            h = np.sum(v_opt)
            
            x = centers[:, 0]
            y = centers[:, 1]
            r = radii
            
            valid, val = _check_validity(x, y, r, w, h)
            if valid and val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy())
        except Exception:
            pass
    
    if best_result is None:
        # Fallback to simple 5x5 grid with uniform cells, ignoring 4 corners
        w, h = 1.0, 1.0
        # This is a rough fallback
        x = np.array([(i % 5) * 0.2 + 0.1 for i in range(25) if i not in [0, 4, 20, 24]])
        y = np.array([(i // 5) * 0.2 + 0.1 for i in range(25) if i not in [0, 4, 20, 24]])
        r = np.full(21, 0.09)
        valid, val = _check_validity(x, y, r, w, h)
        if valid:
            best_result = (x.copy(), y.copy(), r.copy())
    
    if best_result is None:
        return np.array([[0.1, 0.1, 0.05]] * N)
    
    x, y, r = best_result
    
    # Final slight shrink to ensure validity
    for _ in range(5):
        rt = r * (1 - 1e-8)
        w = np.max(x + r)
        h = np.max(y + r)
        if _check_validity(x, y, rt, w, h)[0]:
            r = rt
        else:
            break
            
    return np.column_stack((x, y, r))
