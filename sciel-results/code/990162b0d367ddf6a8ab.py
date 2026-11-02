"""Circle packing in unit square for N=26 circles."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _validate_and_score(circles):
    """Check validity and compute score. circles: (N,3) array [x,y,r]."""
    if circles.shape != (N, 3):
        return None
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(r <= 0):
        return None
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9) or np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return None
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(dist < r[i_idx] + r[j_idx] - 1e-9):
        return None
    return float(np.sum(r))

def _get_delaunay_edges(points):
    """Get edges of Delaunay triangulation."""
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        if not edges:
            # Fallback to all pairs if triangulation fails (shouldn't happen for N=26)
            i_idx, j_idx = np.triu_indices(len(points), k=1)
            return i_idx, j_idx
        edge_list = sorted(edges)
        return np.array([e[0] for e in edge_list]), np.array([e[1] for e in edge_list])
    except Exception:
        i_idx, j_idx = np.triu_indices(len(points), k=1)
        return i_idx, j_idx

def _resolve_positions(x, y, r, time_limit=5.0):
    """
    Given fixed radii r, find positions x, y that minimize overlap and satisfy boundary.
    Uses SLSQP to maximize sum of distances to neighbors (or minimize overlap).
    Actually, we want to find x,y such that dist(i,j) >= r_i + r_j for all neighbors.
    We can minimize the sum of squared violations.
    """
    n = len(r)
    points = np.column_stack([x, y])
    i_idx, j_idx = _get_delaunay_edges(points)
    
    # If the graph changes, we might miss some constraints, but for small moves it's okay.
    # To be safer, we can check all pairs in the final validation, but optimization uses Delaunay.
    
    def objective(params):
        xx = params[:n]
        yy = params[n:2*n]
        
        # Boundary penalty
        bv = np.maximum(0, r - xx) + np.maximum(0, r - (1 - xx)) + \
              np.maximum(0, r - yy) + np.maximum(0, r - (1 - yy))
        
        # Overlap penalty for Delaunay neighbors
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        ov = np.maximum(0, r[i_idx] + r[j_idx] - dist)
        
        # We want to minimize overlap and boundary violations
        return 100.0 * (np.sum(ov * ov) + np.sum(bv * bv))

    def gradient(params):
        xx = params[:n]
        yy = params[n:2*n]
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        
        # Boundary gradients
        gx += np.where(xx < r, 200.0, 0.0) - np.where(1 - xx < r, 200.0, 0.0)
        gy += np.where(yy < r, 200.0, 0.0) - np.where(1 - yy < r, 200.0, 0.0)
        
        # Overlap gradients
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        overlap = r[i_idx] + r[j_idx] - dist
        
        mask = overlap > 0
        if np.any(mask):
            # d/dx ( (r_i+r_j-dist)^2 ) = 2 * overlap * (-d(dist)/dx)
            # d(dist)/dx_i = -dx/dist
            rep = 200.0 * overlap[mask] * (dx[mask] / dist[mask])
            gx[i_idx[mask]] += rep
            gx[j_idx[mask]] -= rep
            
            rep_y = 200.0 * overlap[mask] * (dy[mask] / dist[mask])
            gy[i_idx[mask]] += rep_y
            gy[j_idx[mask]] -= rep_y
            
        return np.concatenate([gx, gy])

    bounds = [(0, 1)] * n + [(0, 1)] * n
    start_time = time.time()
    
    try:
        res = minimize(objective, params=np.concatenate([x, y]), method='SLSQP', 
                       jac=gradient, bounds=bounds, 
                       options={'maxiter': 100, 'ftol': 1e-12, 'disp': False})
        
        if time.time() - start_time > time_limit:
             return x, y # Return original if timeout
            
        new_x = res.x[:n]
        new_y = res.x[n:2*n]
        
        # Clamp to be safe
        new_x = np.clip(new_x, 0, 1)
        new_y = np.clip(new_y, 0, 1)
        
        return new_x, new_y
        
    except Exception:
        return x, y

def _grow_and_relax(circles, time_budget=10.0):
    """
    Iteratively increase radii and resolve overlaps.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    start_time = time.time()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    if best_score is None:
        return circles, 0.0
    
    best_circles = np.column_stack([x, y, r]).copy()
    
    # Try to grow radii
    growth_factor = 1.005 # 0.5% growth per step
    
    while time.time() - start_time < time_budget:
        # Attempt to grow
        new_r = r * growth_factor
        
        # Check if this is even possible (boundary)
        max_r_boundary = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        if np.any(new_r > max_r_boundary):
            # Can't grow uniformly, try smaller growth or break
            growth_factor *= 0.8
            if growth_factor < 1.0001:
                break
            new_r = np.minimum(new_r, max_r_boundary)
        
        # Resolve positions for new radii
        new_x, new_y = _resolve_positions(x, y, new_r, time_limit=2.0)
        
        # Validate
        candidate = np.column_stack([new_x, new_y, new_r])
        score = _validate_and_score(candidate)
        
        if score is not None and score > best_score:
            best_score = score
            best_circles = candidate.copy()
            x, y, r = new_x, new_y, new_r
            # Successful growth, maybe keep growth factor or increase slightly
            # growth_factor = min(1.01, growth_factor * 1.1)
        else:
            # Failed to resolve, shrink growth factor
            growth_factor *= 0.8
            if growth_factor < 1.0001:
                break
                
    return best_circles, best_score

def _initial_hexagonal(n=N):
    """Create a hexagonal grid initial configuration."""
    circles = np.zeros((n, 3))
    cols = 5
    rows = 6
    dx = 1.0 / (cols + 0.5)
    dy = 1.0 / (rows + 0.5)
    r_base = min(dx, dy) * 0.48
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x_offset = 0.5 * dx if row % 2 == 1 else 0
            cx = (col + 0.5 + x_offset) * dx
            cy = (row + 0.5) * dy
            circles[idx, 0] = cx
            circles[idx, 1] = cy
            circles[idx, 2] = r_base
            idx += 1
    return circles

def _initial_grid(n=N):
    """Create a grid initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append((cx, cy, r))
    return np.array(cells[:n])

def _initial_random(n=N, seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    circles[:, 2] = 0.01 + 0.01 * rng.rand(n)
    circles[:, 0] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    circles[:, 1] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    return circles

def construct():
    """Construct optimal circle packing in unit square for N=26 circles."""
    start_time = time.time()
    time_budget = 140  # Keep 10s safety margin before 150s kill
    
    best_circles = None
    best_score = 0.0
    
    # Initializations
    inits = [
        _initial_hexagonal(N),
        _initial_grid(N),
        _initial_random(N, seed=1),
        _initial_random(N, seed=2),
        _initial_random(N, seed=3),
    ]
    
    for init in inits:
        elapsed = time.time() - start_time
        if elapsed > time_budget - 20:
            break
            
        # First, ensure the initial configuration is valid by resolving overlaps
        # We can use a simple relaxation or just start grow-and-relax from a small radius
        # To be safe, let's shrink radii until valid, then grow.
        
        # Actually, _grow_and_relax handles the "grow" part. 
        # We need a valid start.
        # Let's use _resolve_positions to fix the initial state with current radii.
        x0, y0 = init[:, 0], init[:, 1]
        r0 = init[:, 2]
        
        # If initial is invalid, shrink radii
        # Simple loop: shrink all radii by 0.9 until valid
        curr_r = r0.copy()
        for _ in range(10):
            candidate = np.column_stack([x0, y0, curr_r])
            if _validate_and_score(candidate) is not None:
                break
            curr_r *= 0.9
        
        valid_init = np.column_stack([x0, y0, curr_r])
        
        # Now run grow-and-relax
        remaining = time_budget - (time.time() - start_time)
        if remaining < 15:
            continue
            
        improved_circles, score = _grow_and_relax(valid_init, time_budget=min(remaining - 5, 25))
        
        if score > best_score:
            best_score = score
            best_circles = improved_circles
            
        # Optional: Final polish with SLSQP maximizing sum of radii directly
        # This can squeeze out a bit more.
        if best_circles is not None and time.time() - start_time < time_budget - 10:
             # ... (skipping complex direct optimization for now to stay within budget, 
             # grow-and-relax is usually very effective)
             pass

    if best_circles is None:
        return _initial_grid(N)
        
    return best_circles

# EVOLVE-BLOCK-END
