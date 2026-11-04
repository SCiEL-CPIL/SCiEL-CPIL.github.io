"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import random

def _relax_positions(x, y, r, w, h, n_iters=1000, step_size=0.01):
    """
    Relax positions to resolve overlaps and boundary violations.
    Returns updated x, y.
    """
    for _ in range(n_iters):
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Circle-Circle Forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                
                if dist < 1e-8:
                    dx = random.uniform(-1e-6, 1e-6)
                    dy = random.uniform(-1e-6, 1e-6)
                    dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < min_dist:
                    # Overlap: push apart
                    overlap = min_dist - dist
                    force = overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Boundary Forces
        for i in range(N):
            # Left
            if x[i] < r[i]:
                pen = r[i] - x[i]
                fx[i] += pen
            # Right
            if x[i] > w - r[i]:
                pen = x[i] - (w - r[i])
                fx[i] -= pen
            # Bottom
            if y[i] < r[i]:
                pen = r[i] - y[i]
                fy[i] += pen
            # Top
            if y[i] > h - r[i]:
                pen = y[i] - (h - r[i])
                fy[i] -= pen
        
        # Update
        x += step_size * fx
        y += step_size * fy
        
        # Clamp
        for i in range(N):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
            
    return x, y

def _is_valid(x, y, r, w, h, tol=1e-5):
    """Check validity of packing."""
    for i in range(N):
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
        
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist*min_dist - tol:
                return False
    return True

def _solve_packing(w, h, seed=0):
    """
    Solve packing for given w, h.
    Uses iterative radius growth and position relaxation.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize positions randomly
    x = np.random.uniform(0, w, N)
    y = np.random.uniform(0, h, N)
    
    # Initial small radius
    # Estimate max possible radius based on area
    area = w * h
    avg_r = np.sqrt(area / (N * np.pi))
    r = np.full(N, avg_r * 0.1)
    
    # Relax initial positions to remove overlaps
    x, y = _relax_positions(x, y, r, w, h, n_iters=500, step_size=0.05)
    
    # Iterative growth
    growth_factor = 1.01
    max_growth_iters = 200
    current_sum = np.sum(r)
    
    for _ in range(max_growth_iters):
        # Try to grow radii
        r_new = r * growth_factor
        
        # Check if r_new fits in boundaries
        if np.any(r_new > w/2) or np.any(r_new > h/2):
            break
            
        # Relax positions with new radii
        x_new, y_new = _relax_positions(x, y, r_new, w, h, n_iters=500, step_size=0.02)
        
        # Check if valid
        if _is_valid(x_new, y_new, r_new, w, h, tol=1e-4):
            x, y, r = x_new, y_new, r_new
            current_sum = np.sum(r)
        else:
            # If not valid, try smaller growth or break
            # Try a smaller growth step
            growth_factor = 1.005
            r_try = r * growth_factor
            if np.any(r_try > w/2) or np.any(r_try > h/2):
                break
            x_try, y_try = _relax_positions(x, y, r_try, w, h, n_iters=500, step_size=0.02)
            if _is_valid(x_try, y_try, r_try, w, h, tol=1e-4):
                x, y, r = x_try, y_try, r_try
                current_sum = np.sum(r)
            else:
                break
                
    return x, y, r, current_sum

def construct():
    """
    Pack 21 circles in a rectangle of perimeter at most 4.
    """
    best_val = -1
    best_result = None
    
    # Perimeter 2(w+h) = 4 => w+h = 2
    # Try a range of aspect ratios
    # w from 0.5 to 1.5
    aspect_ratios = np.linspace(0.5, 1.5, 20)
    
    for w in aspect_ratios:
        h = 2.0 - w
        
        # Try multiple seeds for this aspect ratio
        for seed in range(5):
            x, y, r, val = _solve_packing(w, h, seed=seed)
            if val > best_val:
                best_val = val
                best_result = np.column_stack([x, y, r])
                
    if best_result is not None:
        return best_result
    
    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(N):
        col = i % side
        row = i // side
        x = (col + 0.5) / side
        y = (row + 0.5) / side
        r = 0.4 / side
        cells.append([x, y, r])
    return np.array(cells)

# EVOLVE-BLOCK-END
