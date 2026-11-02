"""Circle packing in perimeter-4 rectangle: Force-directed + Voronoi expansion + polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = time.time() + 135.0  # Safety margin before 150s limit

def _is_valid(circles, tol=1e-9):
    """Check if circles are valid: inside perimeter-4 rectangle, no overlaps."""
    if circles.shape[0] != N:
        return False
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r <= 0):
        return False
    min_x, max_x = np.min(x - r), np.max(x + r)
    min_y, max_y = np.min(y - r), np.max(y + r)
    w, h = max_x - min_x, max_y - min_y
    if 2.0 * (w + h) > 4.0 + tol:
        return False
    # Check overlaps
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    sum_r = r[:, None] + r[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(dist[mask] < sum_r[mask] - tol):
        return False
    return True

def _value(circles):
    """Sum of radii."""
    return np.sum(circles[:, 2])

def _point_to_segment_dist(px, py, ax, ay, bx, by):
    """Distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-15:
        return np.sqrt((px - ax)**2 + (py - ay)**2)
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / ab2, 0, 1)
    cx, cy = ax + t * abx, ay + t * aby
    return np.sqrt((px - cx)**2 + (py - cy)**2)

def _max_radius_voronoi(circles):
    """
    For each circle, compute the maximum radius it can have given the current
    center positions, considering other circles and the rectangle boundary.
    Returns array of max radii.
    """
    x, y = circles[:, 0], circles[:, 1]
    n = len(x)
    max_r = np.zeros(n)
    for i in range(n):
        r_i = 1e10
        # Distance to other circles
        for j in range(n):
            if i == j:
                continue
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            r_i = min(r_i, d / 2.0)
        # Distance to rectangle edges (we need to figure out the rectangle)
        # The rectangle is the bounding box of all circles
        # For now, use a conservative estimate
        max_r[i] = r_i
    return max_r

def _force_relax(circles, n_iters=500, dt=0.005, temp=1.0):
    """
    Force-directed relaxation: circles repel each other and are pushed
    toward center of mass, while being contained in a rectangle.
    """
    circles = circles.copy()
    x, y, r = circles[:, 0].copy(), circles[:, 1].copy(), circles[:, 2].copy()
    
    for it in range(n_iters):
        if time.time() > DEADLINE - 15:
            break
        forces_x = np.zeros(N)
        forces_y = np.zeros(N)
        
        # Repulsion between circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-10:
                    dx, dy = 1e-5, 1e-5
                    d = np.sqrt(dx * dx + dy * dy)
                overlap = (r[i] + r[j]) - d
                if overlap > 0:
                    # Strong repulsion when overlapping
                    f = overlap * 10.0 + 0.1
                    fx = f * dx / d
                    fy = f * dy / d
                    forces_x[i] += fx
                    forces_y[i] += fy
                    forces_x[j] -= fx
                    forces_y[j] -= fy
        
        # Containment: push circles toward center if outside a bounding region
        # Use current bounding box as reference
        min_x, max_x = np.min(x - r), np.max(x + r)
        min_y, max_y = np.min(y - r), np.max(y + r)
        w, h = max_x - min_x, max_y - min_y
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        
        for i in range(N):
            # If circle extends beyond a target box, push it in
            target_w, target_h = w, h
            if x[i] - r[i] < min_x + 0.01:
                forces_x[i] += 2.0
            if x[i] + r[i] > max_x - 0.01:
                forces_x[i] -= 2.0
            if y[i] - r[i] < min_y + 0.01:
                forces_y[i] += 2.0
            if y[i] + r[i] > max_y - 0.01:
                forces_y[j if False else i] -= 2.0
        
        # Apply forces with cooling
        x += dt * forces_x
        y += dt * forces_y
        
        circles[:, 0] = x
        circles[:, 1] = y
    
    return circles

def _expand_radii(circles, n_iters=100):
    """
    Iteratively expand radii as much as possible given current positions.
    """
    circles = circles.copy()
    x, y = circles[:, 0].copy(), circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    for it in range(n_iters):
        if time.time() > DEADLINE - 15:
            break
        changed = False
        for i in range(N):
            max_r = 1e10
            # Limit by other circles
            for j in range(N):
                if i == j:
                    continue
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                max_r = min(max_r, d / 2.0 - 1e-10)
            # Limit by rectangle (use current bounding box)
            min_x = np.min(x - r)
            max_x = np.max(x + r)
            min_y = np.min(y - r)
            max_y = np.max(y + r)
            # Distance to each edge of bounding box
            max_r = min(max_r, x[i] - min_x, max_x - x[i], y[i] - min_y, max_y - y[i])
            max_r = max(max_r, r[i] * 0.5)  # Don't shrink
            if max_r > r[i] + 1e-10:
                r[i] = max_r
                changed = True
        if not changed:
            break
    
    circles[:, 2] = r
    return circles

def _slsqp_refine(circles, max_iter=200):
    """
    Use SLSQP to refine the configuration, maximizing sum of radii.
    """
    x0 = circles.flatten()
    
    def objective(v):
        c = v.reshape(-1, 3)
        if not _is_valid(c):
            return 1e10
        return -_value(c)
    
    def constraint_perimeter(v):
        c = v.reshape(-1, 3)
        x, y, r = c[:, 0], c[:, 1], c[:, 2]
        w = np.max(x + r) - np.min(x - r)
        h = np.max(y + r) - np.min(y - r)
        return 4.0 - 2.0 * (w + h)
    
    constraints = [{'type': 'ineq', 'fun': constraint_perimeter}]
    
    bounds = [(0, 2), (0, 2), (0.001, 1.0)] * N
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                         constraints=constraints,
                         options={'maxiter': max_iter, 'ftol': 1e-12})
        if result.success or _is_valid(result.x.reshape(-1, 3)):
            return result.x.reshape(-1, 3)
    except Exception:
        pass
    return circles

def _polish_result(circles):
    """Final polish: small perturbations + local optimization."""
    best = circles.copy()
    best_val = _value(best) if _is_valid(best) else 0
    
    for attempt in range(5):
        if time.time() > DEADLINE - 15:
            break
        # Small random perturbation
        perturbed = best.copy()
        noise = np.random.normal(0, 0.01, (N, 2))
        perturbed[:, :2] += noise
        # Re-expand radii
        expanded = _expand_radii(perturbed, n_iters=50)
        if _is_valid(expanded):
            val = _value(expanded)
            if val > best_val:
                best = expanded
                best_val = val
    
    return best

def _initial_grid():
    """Create a well-distributed grid initialization."""
    # 5x5 grid, take 21
    cells = []
    for i in range(5):
        for j in range(5):
            if len(cells) >= N:
                break
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 5.0
            r = 0.5 / 5.0 - 0.005
            cells.append([x, y, r])
    return np.array(cells[:N])

def _initial_random():
    """Random initialization in unit square."""
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.random.uniform(0.05, 0.1, N)
    return np.column_stack((x, y, r))

def construct():
    start_time = time.time()
    best_circles = None
    best_value = 0.0
    
    # 1. Grid initialization + force relax + expand + SLSQP
    init = _initial_grid()
    relaxed = _force_relax(init, n_iters=300, dt=0.003)
    expanded = _expand_radii(relaxed, n_iters=100)
    if _is_valid(expanded):
        val = _value(expanded)
        if val > best_value:
            best_value = val
            best_circles = expanded.copy()
    
    if time.time() < DEADLINE - 30:
        refined = _slsqp_refine(expanded, max_iter=100)
        if _is_valid(refined):
            val = _value(refined)
            if val > best_value:
                best_value = val
                best_circles = refined.copy()
    
    # 2. Multiple random initializations
    for attempt in range(8):
        if time.time() > DEADLINE - 30:
            break
        
        init = _initial_random()
        relaxed = _force_relax(init, n_iters=200, dt=0.003)
        expanded = _expand_radii(relaxed, n_iters=100)
        if _is_valid(expanded):
            val = _value(expanded)
            if val > best_value:
                best_value = val
                best_circles = expanded.copy()
        
        if time.time() < DEADLINE - 30:
            refined = _slsqp_refine(expanded, max_iter=80)
            if _is_valid(refined):
                val = _value(refined)
                if val > best_value:
                    best_value = val
                    best_circles = refined.copy()
    
    # 3. Polish the best result
    if best_circles is not None and time.time() < DEADLINE - 15:
        polished = _polish_result(best_circles)
        if _is_valid(polished):
            val = _value(polished)
            if val > best_value:
                best_value = val
                best_circles = polished
    
    if best_circles is None:
        best_circles = _initial_grid()
    
    return best_circles
