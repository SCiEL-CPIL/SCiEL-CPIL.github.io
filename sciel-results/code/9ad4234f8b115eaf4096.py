"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _check_valid(circles, W, H, tol=1e-6):
    """Check if circles are valid (within bounds, non-overlapping)."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > W + tol or y - r < -tol or y + r > H + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - tol:
                return False
    return True

def _compute_value(circles, W, H):
    """Compute the objective: sum of radii, if valid."""
    if not _check_valid(circles, W, H):
        return 0.0
    return np.sum(circles[:, 2])

def _make_grid_packing(m, n, W, H, n_circles):
    """Create a grid packing with m rows and n columns, placing n_circles circles."""
    # Place circles in a grid pattern
    circles = []
    for i in range(m):
        for j in range(n):
            if len(circles) >= n_circles:
                break
            x = (j + 0.5) * W / n
            y = (i + 0.5) * H / m
            r = min(W / (2*n), H / (2*m)) * 0.99
            circles.append([x, y, r])
        if len(circles) >= n_circles:
            break
    return np.array(circles)

def _optimize_packing(circles, W, H, max_iters=2000, tol=1e-8):
    """Optimize packing using coordinate descent."""
    circles = circles.copy().astype(float)
    n = len(circles)
    
    for it in range(max_iters):
        improved = False
        for i in range(n):
            x, y, r = circles[i]
            
            # Try to increase radius
            max_r = min(x, W - x, y, H - y)  # Distance to walls
            for j in range(n):
                if i == j:
                    continue
                xj, yj, rj = circles[j]
                dist = np.sqrt((x - xj)**2 + (y - yj)**2)
                if dist > 0:
                    max_r = min(max_r, dist - rj)
            new_r = max(0, max_r) * 0.999
            if new_r > r + tol:
                circles[i, 2] = new_r
                improved = True
            
            # Try moving to increase radius further
            # For simplicity, just do radius optimization for now
        if not improved:
            break
    
    return circles

def _try_aspect_ratio(W, H, n_circles=21):
    """Try packing for given aspect ratio."""
    if W + H != 2.0:
        scale = 2.0 / (W + H)
        W *= scale
        H *= scale
    
    best_circles = None
    best_value = 0.0
    
    # Try different grid configurations
    for m in range(1, 8):
        for n in range(1, 8):
            if m * n < n_circles:
                continue
            if m * n > n_circles * 2:  # Don't waste too much
                continue
            
            circles = _make_grid_packing(m, n, W, H, n_circles)
            if len(circles) < n_circles:
                continue
            
            # Optimize
            circles_opt = _optimize_packing(circles, W, H)
            value = _compute_value(circles_opt, W, H)
            
            if value > best_value:
                best_value = value
                best_circles = circles_opt
    
    return best_circles, best_value

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_circles = None
    best_value = 0.0
    best_W, best_H = 1.0, 1.0
    
    # Try various aspect ratios
    aspect_ratios = [
        (1.0, 1.0),      # Square
        (1.5, 0.5),      # Wide
        (0.5, 1.5),      # Tall
        (2.0, 0.0),      # Very wide (degenerate)
        (0.0, 2.0),      # Very tall (degenerate)
        (1.2, 0.8),
        (0.8, 1.2),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.8, 0.2),
        (0.2, 1.8),
        (1.6, 0.4),
        (0.4, 1.6),
        (1.3, 0.7),
        (0.7, 1.3),
    ]
    
    for W, H in aspect_ratios:
        if time.time() > deadline:
            break
        
        circles, value = _try_aspect_ratio(W, H, N)
        
        if circles is not None and value > best_value:
            best_value = value
            best_circles = circles
            best_W, best_H = W, H
    
    # If we didn't find anything, use the seed
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0
    
    return best_circles

# EVOLVE-BLOCK-END
