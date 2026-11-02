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

def _softplus(v, temp=0.01):
    v_clipped = np.clip(v / temp, -50, 50)
    return temp * np.log1p(np.exp(v_clipped))

def _softplus_grad(v, temp=0.01):
    v_clipped = np.clip(v / temp, -50, 50)
    return 1.0 / (1.0 + np.exp(-v_clipped))

def _hex_initial(n=N, seed=0):
    """
    Generate a hexagonal initial configuration.
    This creates a dense packing that is valid and provides a good starting point.
    """
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    
    # Approximate radius for 26 circles
    # Area per circle ~ 1/26. r ~ sqrt(1/(26*pi)) ~ 0.11
    # But packing efficiency is ~0.9, so r is smaller.
    # Let's try to fit a hex grid.
    # Rows of circles.
    # If we have 5 columns and 6 rows, we have 30 circles. We need 26.
    # Let's calculate spacing based on a target radius.
    
    r_target = 0.09
    dx = 2 * r_target * 1.01  # Slightly more than diameter
    dy = dx * np.sqrt(3) / 2
    
    points = []
    # Generate a grid large enough, then center it
    rows = 6
    cols = 5
    
    x_offset = 0.5 - (cols * dx) / 2.0
    y_offset = 0.5 - (rows * dy) / 2.0
    
    for row in range(rows):
        for col in range(cols):
            cx = x_offset + col * dx + (dx / 2 if row % 2 else 0)
            cy = y_offset + row * dy
            points.append([cx, cy, r_target])
            
    points = np.array(points[:n])
    
    # Add small noise to break symmetry
    points[:, 0] += 0.005 * rng.randn(n)
    points[:, 1] += 0.005 * rng.randn(n)
    
    # Clamp to valid range
    points[:, 0] = np.clip(points[:, 0], 0.01, 0.99)
    points[:, 1] = np.clip(points[:, 1], 0.01, 0.99)
    
    # Ensure radii are valid for the clamped positions
    max_r = np.minimum(np.minimum(points[:, 0], 1 - points[:, 0]), np.minimum(points[:, 1], 1 - points[:, 1]))
    points[:, 2] = np.minimum(points[:, 2], max_r)
    points[:, 2] = np.maximum(points[:, 2], 0.01)
    
    return points

def _corner_initial(n=N, seed=0):
    """
    Initialize with large circles in corners and smaller ones in the center.
    """
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    
    # 4 corners
    corners = [(0.05, 0.05), (0.95, 0.05), (0.05, 0.95), (0.95, 0.95)]
    corner_r = 0.15
    
    idx = 0
    for cx, cy in corners:
        if idx >= n: break
        circles[idx] = [cx, cy, corner_r]
        idx += 1
        
    # Fill rest with random small circles in the middle
    if idx < n:
        r_fill = 0.06
        # Random placement in the central region
        for i in range(idx, n):
            # Sample from a distribution that avoids corners
            cx = 0.2 + 0.6 * rng.rand()
            cy = 0.2 + 0.6 * rng.rand()
            # Simple rejection sampling to avoid overlap with corners (roughly)
            # Just place and let optimizer fix
            circles[i] = [cx, cy, r_fill]
            
    return circles

def _lbfgs_delaunay(circles, time_limit=10.0, seed=0):
    """
    Local optimization using L-BFGS-B and Delaunay triangulation for pair constraints.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    start_time = time.time()
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # Cache Delaunay edges
    def _get_edges(cx, cy):
        try:
            tri = Delaunay(np.column_stack([cx, cy]))
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        if a > b: a, b = b, a
                        edges.add((a, b))
            if not edges: return i_idx, j_idx
            edge_list = sorted(edges)
            return np.array([e[0] for e in edge_list]), np.array([e[1] for e in edge_list])
        except:
            return i_idx, j_idx
            
    cached_di, cached_dj = _get_edges(x, y)
    
    temp = 0.02
    
    def objective(p):
        if time.time() - start_time > time_limit: return 1e10
        xx, yy, rr = p[:n], p[n:2*n], p[2*n:3*n]
        
        # Boundary violations
        bv = _softplus(rr - xx, temp) + _softplus(rr - (1-xx), temp) + \
             _softplus(rr - yy, temp) + _softplus(rr - (1-yy), temp)
        
        # Pair violations (only Delaunay neighbors)
        dx = xx[cached_di] - xx[cached_dj]
        dy = yy[cached_di] - yy[cached_dj]
        dist = np.sqrt(dx*dx + dy*dy)
        dist = np.maximum(dist, 1e-12)
        ov = _softplus(rr[cached_di] + rr[cached_dj] - dist, temp)
        
        return -np.sum(rr) + 100.0 * (np.sum(bv) + np.sum(ov))
        
    def gradient(p):
        if time.time() - start_time > time_limit: return np.zeros_like(p)
        xx, yy, rr = p[:n], p[n:2*n], p[2*n:3*n]
        
        gx, gy, gr = np.zeros(n), np.zeros(n), np.zeros(n)
        
        # Boundary grads
        gx += 100.0 * (_softplus_grad(rr - xx, temp) - _softplus_grad(rr - (1-xx), temp))
        gy += 100.0 * (_softplus_grad(rr - yy, temp) - _softplus_grad(rr - (1-yy), temp))
        gr += 100.0 * (_softplus_grad(rr - xx, temp) + _softplus_grad(rr - (1-xx), temp) + 
                       _softplus_grad(rr - yy, temp) + _softplus_grad(rr - (1-yy), temp))
        
        # Pair grads
        dx = xx[cached_di] - xx[cached_dj]
        dy = yy[cached_di] - yy[cached_dj]
        dist = np.sqrt(dx*dx + dy*dy)
        dist = np.maximum(dist, 1e-12)
        overlap = rr[cached_di] + rr[cached_dj] - dist
        
        mask = overlap > -0.02
        if np.any(mask):
            rep = 100.0 * _softplus_grad(overlap[mask], temp) / dist[mask]
            gx[cached_di[mask]] -= rep * dx[mask] / dist[mask]
            gy[cached_di[mask]] -= rep * dy[mask] / dist[mask]
            gx[cached_dj[mask]] += rep * dx[mask] / dist[mask]
            gy[cached_dj[mask]] += rep * dy[mask] / dist[mask]
            gr[cached_di[mask]] += 100.0 * _softplus_grad(overlap[mask], temp)
            gr[cached_dj[mask]] += 100.0 * _softplus_grad(overlap[mask], temp)
            
        gr -= 1.0
        return np.concatenate([gx, gy, gr])
        
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.001, 0.5)] * n
    
    # Update Delaunay cache periodically is hard inside L-BFGS callback, 
    # so we run a few iterations, update cache, and run again.
    
    best_circles = circles.copy()
    best_score = _validate_and_score(circles) or 0.0
    
    current_params = np.concatenate([x, y, r])
    
    # Run 3 rounds of L-BFGS with cache updates
    for round in range(3):
        if time.time() - start_time > time_limit - 1: break
        
        cached_di, cached_dj = _get_edges(current_params[:n], current_params[n:2*n])
        
        try:
            res = minimize(objective, current_params, method='L-BFGS-B', jac=gradient,
                           bounds=bounds, options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            current_params = res.x
        except:
            pass
            
        # Validate and update best
        new_x = np.clip(current_params[:n], 0, 1)
        new_y = np.clip(current_params[n:2*n], 0, 1)
        new_r = np.maximum(current_params[2*n:3*n], 0.001)
        max_r = np.minimum(np.minimum(new_x, 1-new_x), np.minimum(new_y, 1-new_y))
        new_r = np.minimum(new_r, np.maximum(max_r, 0.001))
        
        new_circles = np.column_stack([new_x, new_y, new_r])
        new_score = _validate_and_score(new_circles)
        if new_score is not None and new_score > best_score:
            best_score = new_score
            best_circles = new_circles
    
    return best_circles, best_score

def _sim_anneal(circles, time_limit=15.0, seed=0):
    """
    Simulated Annealing with softplus penalty.
    """
    rng = np.random.RandomState(seed)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    start_time = time.time()
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def _penalty(cx, cy, cr, temp=0.02):
        bv = _softplus(cr - cx, temp) + _softplus(cr - (1-cx), temp) + \
             _softplus(cr - cy, temp) + _softplus(cr - (1-cy), temp)
        dx = cx[i_idx] - cx[j_idx]
        dy = cy[i_idx] - cy[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        dist = np.maximum(dist, 1e-12)
        ov = _softplus(cr[i_idx] + cr[j_idx] - dist, temp)
        return -np.sum(cr) + 100.0 * (np.sum(bv) + np.sum(ov))
        
    current_energy = _penalty(x, y, r)
    best_circles = circles.copy()
    best_score = _validate_and_score(circles) or 0.0
    best_energy = current_energy
    
    t = 0.05
    t_min = 0.001
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        i = rng.randint(n)
        move_type = rng.rand()
        
        new_x, new_y, new_r = x.copy(), y.copy(), r.copy()
        
        if move_type < 0.5:
            # Move position
            new_x[i] += 0.02 * rng.randn()
            new_y[i] += 0.02 * rng.randn()
        elif move_type < 0.8:
            # Change radius
            new_r[i] *= (1 + 0.1 * rng.randn())
            new_r[i] = np.clip(new_r[i], 0.01, 0.5)
        else:
            # Swap radii (helps with heterogeneity)
            j = rng.randint(n)
            if j != i:
                new_r[i], new_r[j] = new_r[j], new_r[i]
                
        # Clamp positions to box roughly
        new_x[i] = np.clip(new_x[i], 0.0, 1.0)
        new_y[i] = np.clip(new_y[i], 0.0, 1.0)
        
        new_energy = _penalty(new_x, new_y, new_r)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.rand() < np.exp(-delta / max(t, 1e-10)):
            x, y, r = new_x, new_y, new_r
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                # Validate strict score
                s = _validate_and_score(np.column_stack([x, y, r]))
                if s is not None and s > best_score:
                    best_score = s
                    best_circles = np.column_stack([x, y, r]).copy()
                    
        t *= cooling
        if t < t_min: t = t_min
        
    return best_circles, best_score

def construct():
    """
    Main entry point.
    """
    start_time = time.time()
    time_budget = 140.0  # Safety margin
    
    best_circles = None
    best_score = 0.0
    
    # Initializations
    inits = [
        _hex_initial(N, seed=1),
        _hex_initial(N, seed=2),
        _corner_initial(N, seed=3),
        _hex_initial(N, seed=4),
    ]
    
    for i, init in enumerate(inits):
        if time.time() - start_time > time_budget - 20:
            break
            
        # 1. Simulated Annealing
        sa_time = min(10.0, (time_budget - (time.time() - start_time)) * 0.5)
        if sa_time > 2:
            sa_circles, sa_score = _sim_anneal(init, time_limit=sa_time, seed=i*10+1)
            if sa_score > best_score:
                best_score = sa_score
                best_circles = sa_circles
                
        # 2. L-BFGS Polish
        if best_circles is not None:
            lbfgs_time = min(5.0, (time_budget - (time.time() - start_time)) * 0.3)
            if lbfgs_time > 2:
                lbfgs_circles, lbfgs_score = _lbfgs_delaunay(best_circles, time_limit=lbfgs_time, seed=i*10+2)
                if lbfgs_score > best_score:
                    best_score = lbfgs_score
                    best_circles = lbfgs_circles
                    
    # Final global polish if time remains
    if best_circles is not None:
        remaining = time_budget - (time.time() - start_time)
        if remaining > 5:
            # One more SA pass on the best found
            sa_time = min(remaining * 0.5, 10.0)
            sa_circles, sa_score = _sim_anneal(best_circles, time_limit=sa_time, seed=99)
            if sa_score > best_score:
                best_score = sa_score
                best_circles = sa_circles
                
            # One more L-BFGS pass
            remaining = time_budget - (time.time() - start_time)
            if remaining > 3:
                lbfgs_time = min(remaining * 0.8, 5.0)
                lbfgs_circles, lbfgs_score = _lbfgs_delaunay(best_circles, time_limit=lbfgs_time, seed=100)
                if lbfgs_score > best_score:
                    best_score = lbfgs_score
                    best_circles = lbfgs_circles
    
    if best_circles is None:
        return _hex_initial(N, seed=0)
        
    return best_circles

# EVOLVE-BLOCK-END
