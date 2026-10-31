"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _validate(circles, eps=1e-9):
    """Check if circles are valid: disjoint interiors, inside bounding rect with perimeter <= 4."""
    if len(circles) < N:
        return False
    circles = np.array(circles)
    cx, cy, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r < 0):
        return False
    # Check pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((cx[i] - cx[j])**2 + (cy[i] - cy[j])**2)
            if dist < r[i] + r[j] - eps:
                return False
    # Bounding box
    xmin = np.min(cx - r)
    xmax = np.max(cx + r)
    ymin = np.min(cy - r)
    ymax = np.max(cy + r)
    width = xmax - xmin
    height = ymax - ymin
    perimeter = 2 * (width + height)
    if perimeter > 4 + eps:
        return False
    return True


def _sum_radii(circles):
    return np.sum(circles[:, 2])


def _grid_init(n_circles, aspect_ratio):
    """Generate initial grid packing for n circles in a rectangle of given aspect ratio."""
    # Rectangle with perimeter 4: 2*(w+h) = 4 => w+h = 2
    # aspect_ratio = w/h
    # w = aspect_ratio * h
    # aspect_ratio*h + h = 2 => h = 2/(1+aspect_ratio), w = 2*aspect_ratio/(1+aspect_ratio)
    h = 2.0 / (1.0 + aspect_ratio)
    w = aspect_ratio * h
    
    # Arrange in grid
    # Find grid dimensions
    best_score = -1
    best_grid = None
    for rows in range(1, n_circles + 1):
        cols = int(np.ceil(n_circles / rows))
        if rows * cols < n_circles:
            continue
        # Grid cells
        cw = w / cols
        ch = h / rows
        cells = []
        for i in range(rows):
            for j in range(cols):
                if len(cells) >= n_circles:
                    break
                x = (j + 0.5) * cw
                y = (i + 0.5) * ch
                r = min(cw, ch) / 2.0 - 1e-4
                cells.append([x, y, r])
            if len(cells) >= n_circles:
                break
        if len(cells) >= n_circles:
            cells = cells[:n_circles]
            score = sum(c[2] for c in cells)
            if score > best_score:
                best_score = score
                best_grid = cells
    if best_grid is None:
        # Fallback
        h = 2.0 / (1.0 + aspect_ratio)
        w = aspect_ratio * h
        best_grid = [[w/2, h/2, min(w,h)/2 - 1e-4]] + [[w/2, h/2, 0.0]] * (n_circles - 1)
    return np.array(best_grid)


def _force_relax(circles, aspect_ratio, iterations=500, dt=0.01):
    """Simple force-directed relaxation to improve packing."""
    circles = np.array(circles, dtype=float).copy()
    h = 2.0 / (1.0 + aspect_ratio)
    w = aspect_ratio * h
    
    for it in range(iterations):
        forces = np.zeros_like(circles)
        # Repulsion between circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = circles[i, 2] + circles[j, 2]
                if dist < min_dist and dist > 1e-10:
                    # Push apart
                    f = (min_dist - dist) * 10.0
                    fx = f * dx / dist
                    fy = f * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        # Attraction to keep circles inside and centered
        for i in range(N):
            # Center attraction
            forces[i, 0] += -circles[i, 0] * 0.1
            forces[i, 1] += -circles[i, 1] * 0.1
            # Wall repulsion
            if circles[i, 0] - circles[i, 2] < 0:
                forces[i, 0] += (0 - (circles[i, 0] - circles[i, 2])) * 5.0
            if circles[i, 0] + circles[i, 2] > w:
                forces[i, 0] -= (circles[i, 0] + circles[i, 2] - w) * 5.0
            if circles[i, 1] - circles[i, 2] < 0:
                forces[i, 1] += (0 - (circles[i, 1] - circles[i, 2])) * 5.0
            if circles[i, 1] + circles[i, 2] > h:
                forces[i, 1] -= (circles[i, 1] + circles[i, 2] - h) * 5.0
        
        circles[:, :2] += dt * forces
        # Clamp radii to non-negative
        circles[:, 2] = np.maximum(circles[:, 2], 0.0)
    
    return circles


def _optimize_aspect(circles, aspect_ratio, time_budget=10.0):
    """Try to optimize the aspect ratio and circle positions using SLSQP."""
    start_time = time.time()
    
    # Variables: cx, cy, r for each circle, plus aspect ratio
    # We'll optimize positions and radii, keeping aspect fixed for now
    x0 = circles.flatten().tolist()
    
    def objective(x):
        if time.time() - start_time > time_budget:
            return 1e10
        x = np.array(x)
        cx = x[0:N]
        cy = x[N:2*N]
        r = x[2*N:3*N]
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist)**2 * 1000
        # Penalty for going out of bounds
        h = 2.0 / (1.0 + aspect_ratio)
        w = aspect_ratio * h
        for i in range(N):
            if cx[i] - r[i] < 0:
                penalty += (cx[i] - r[i])**2 * 1000
            if cx[i] + r[i] > w:
                penalty += (cx[i] + r[i] - w)**2 * 1000
            if cy[i] - r[i] < 0:
                penalty += (cy[i] - r[i])**2 * 1000
            if cy[i] + r[i] > h:
                penalty += (cy[i] + r[i] - h)**2 * 1000
        # Maximize sum of radii, so minimize negative
        return -np.sum(r) + penalty
    
    res = minimize(objective, x0, method='SLSQP', options={'maxiter': 200, 'ftol': 1e-8})
    if time.time() - start_time > time_budget:
        return circles
    
    x = res.x
    cx = x[0:N]
    cy = x[N:2*N]
    r = x[2*N:3*N]
    new_circles = np.column_stack([cx, cy, r])
    
    # Validate
    if _validate(new_circles):
        return new_circles
    return circles


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s
    
    best_circles = None
    best_score = -np.inf
    
    # Try different aspect ratios
    aspect_ratios = [0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # Initialize with grid
        circles = _grid_init(N, ar)
        
        # Force relaxation
        circles = _force_relax(circles, ar, iterations=300, dt=0.005)
        
        # Check validity and score
        if _validate(circles):
            score = _sum_radii(circles)
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
        
        # Try SLSQP optimization if time permits
        if time.time() < deadline - 15.0:
            optimized = _optimize_aspect(circles, ar, time_budget=10.0)
            if _validate(optimized):
                score = _sum_radii(optimized)
                if score > best_score:
                    best_score = score
                    best_circles = optimized.copy()
    
    # Final fallback
    if best_circles is None:
        best_circles = _grid_init(N, 1.0)
    
    return best_circles
# EVOLVE-BLOCK-END
