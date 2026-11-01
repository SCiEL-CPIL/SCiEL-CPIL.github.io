"""Circle packing: 50 equal circles in unit square [-1/2, 1/2]^2.
Maximize the common radius via simulated annealing with hexagonal init."""

import numpy as np
import time

N = 50

def _compute_radius(pts):
    """Compute the common radius for a set of points.
    r = min(min_{i<j} |c_i - c_j|/2, min_i min_k (1/2 - |c_ik|))
    """
    n = len(pts)
    # Pairwise distances
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min() / 2.0
    # Wall distances
    wall = 0.5 - np.abs(pts).max(axis=1)
    min_wall = wall.min()
    return min(min_pair, min_wall)

def _hex_init(n):
    """Generate a hexagonal lattice initial placement for n circles in [-1/2, 1/2]^2."""
    best_pts = None
    best_r = 0.0
    
    # Try several grid configurations with hexagonal offset
    for cols in range(6, 10):
        rows = int(np.ceil(n / cols))
        x_spacing = 1.0 / cols
        y_spacing = 1.0 / rows
        pts = []
        for r in range(rows):
            for c in range(cols):
                if len(pts) >= n:
                    break
                x = (c + 0.5) * x_spacing - 0.5
                y = (r + 0.5) * y_spacing - 0.5
                if r % 2 == 1:
                    x += x_spacing * 0.5
                x = np.clip(x, -0.49, 0.49)
                y = np.clip(y, -0.49, 0.49)
                pts.append([x, y])
        pts = np.array(pts[:n])
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial placement: hexagonal
    pts = _hex_init(N)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    current_r = best_r
    
    # Simulated annealing
    T = 0.01
    T_end = 1e-08
    step_size = 0.035
    
    rng = np.random.default_rng(42)
    
    it = 0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Propose perturbation
        idx = rng.integers(0, N)
        dx = rng.uniform(-step_size, step_size)
        dy = rng.uniform(-step_size, step_size)
        
        old_x = pts[idx, 0]
        old_y = pts[idx, 1]
        new_x = np.clip(old_x + dx, -0.5, 0.5)
        new_y = np.clip(old_y + dy, -0.5, 0.5)
        
        pts[idx, 0] = new_x
        pts[idx, 1] = new_y
        
        new_r = _compute_radius(pts)
        
        # Accept if better, or with probability exp((new_r - current_r) / T)
        if new_r >= current_r:
            # Accept
            current_r = new_r
        else:
            delta = new_r - current_r  # negative
            prob = np.exp(delta / T)
            if rng.random() > prob:
                # Reject
                pts[idx, 0] = old_x
                pts[idx, 1] = old_y
                current_r = _compute_radius(pts)
                no_improve += 1
            else:
                # Accept worse move
                current_r = new_r
                no_improve += 1
        
        if current_r > best_r:
            best_r = current_r
            best_pts = pts.copy()
            no_improve = 0
        
        it += 1
        
        # Cool down
        if it % 100 == 0:
            T *= 0.995
            if T < T_end:
                T = T_end
        
        # Periodically reheat if stuck
        if no_improve > max_no_improve:
            T = max(T, 0.005)
            no_improve = 0
        
        # Check deadline
        if time.time() > deadline - 5.0:
            break
    
    # Final polish: smaller steps
    if time.time() < deadline - 1.0:
        T = 1e-06
        step_size = 0.005
        for _ in range(20000):
            if time.time() > deadline - 1.0:
                break
            idx = rng.integers(0, N)
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            
            old_x = pts[idx, 0]
            old_y = pts[idx, 1]
            new_x = np.clip(old_x + dx, -0.5, 0.5)
            new_y = np.clip(old_y + dy, -0.5, 0.5)
            
            pts[idx, 0] = new_x
            pts[idx, 1] = new_y
            new_r = _compute_radius(pts)
            
            if new_r < current_r:
                prob = np.exp((new_r - current_r) / T)
                if rng.random() > prob:
                    pts[idx, 0] = old_x
                    pts[idx, 1] = old_y
                    current_r = _compute_radius(pts)
            else:
                current_r = new_r
            
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
    
    # Ensure final points are strictly inside
    best_pts = np.clip(best_pts, -0.499999, 0.499999)
    
    return best_pts
