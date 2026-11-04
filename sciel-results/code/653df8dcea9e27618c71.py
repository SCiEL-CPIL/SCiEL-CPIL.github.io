"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return sum(c[2] for c in circles)

def _force_directed_layout(w, h, init_pos, max_iter=2000, tol=1e-10):
    """
    Force-directed layout: circles repel each other when overlapping,
    and are attracted to grow as much as possible while staying in bounds.
    """
    pos = init_pos.copy()
    # Start with small radii
    max_possible = min(w, h) / 2
    rad = np.full(N, 0.05 * max_possible)
    
    # Adaptive step size
    step = 0.01
    
    for it in range(max_iter):
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        # Repulsion between overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = rad[i] + rad[j]
                if d < min_d and d > 1e-12:
                    # Repulsive force proportional to overlap
                    overlap = min_d - d
                    nx, ny = dx / d, dy / d
                    force = overlap * 5.0
                    fx[i] += nx * force
                    fy[i] += ny * force
                    fx[j] -= nx * force
                    fy[j] -= ny * force
                    # Slightly shrink both circles
                    fr[i] -= overlap * 0.5
                    fr[j] -= overlap * 0.5
                elif d <= 1e-12:
                    # Random push
                    angle = np.random.uniform(0, 2 * np.pi)
                    fx[i] += np.cos(angle) * 0.1
                    fy[i] += np.sin(angle) * 0.1
                    fx[j] -= np.cos(angle) * 0.1
                    fy[j] -= np.sin(angle) * 0.1
        
        # Boundary forces: push circles inward if they touch walls
        for i in range(N):
            # Left wall
            if pos[i, 0] - rad[i] < 0:
                fx[i] += (rad[i] - pos[i, 0]) * 10.0
            # Right wall
            if pos[i, 0] + rad[i] > w:
                fx[i] -= (pos[i, 0] + rad[i] - w) * 10.0
            # Bottom wall
            if pos[i, 1] - rad[i] < 0:
                fy[i] += (rad[i] - pos[i, 1]) * 10.0
            # Top wall
            if pos[i, 1] + rad[i] > h:
                fy[i] -= (pos[i, 1] + rad[i] - h) * 10.0
            
            # Growth force: try to increase radius
            # Limited by distance to nearest boundary and other circles
            max_r_boundary = min(pos[i, 0], w - pos[i, 0], pos[i, 1], h - pos[i, 1])
            if rad[i] < max_r_boundary:
                fr[i] += 0.01  # Gentle growth push
        
        # Update positions
        pos[:, 0] += step * fx
        pos[:, 1] += step * fy
        rad += step * fr
        
        # Clamp radii to valid range
        for i in range(N):
            rad[i] = min(rad[i], pos[i, 0], w - pos[i, 0], pos[i, 1], h - pos[i, 1])
            rad[i] = max(rad[i], 1e-6)
        
        # Clamp positions to bounds
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
            pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
        
        # Check convergence
        total_force = np.sqrt(np.sum(fx**2) + np.sum(fy**2) + np.sum(fr**2))
        if total_force < tol:
            break
        
        # Decay step size
        step *= 0.999
    
    return np.column_stack([pos, rad])

def _hexagonal_init(w, h, nc, nr):
    """Generate hexagonal-like grid positions."""
    positions = []
    row_spacing = h / nr
    col_spacing = w / nc
    for j in range(nr):
        y = (j + 0.5) * row_spacing
        offset = 0.0 if j % 2 == 0 else 0.5 * col_spacing
        for i in range(nc):
            if len(positions) >= N:
                break
            x = (i + 0.5) * col_spacing + offset
            x = min(max(x, 0), w)
            positions.append([x, y])
    while len(positions) < N:
        idx = len(positions)
        x = w / 2 + (idx - N / 2) * w / N * 0.3
        y = h / 2 + (idx - N / 2) * h / N * 0.3
        positions.append([x, y])
    return np.array(positions[:N])

def _grid_init(w, h, nc, nr):
    """Generate initial circle positions in a grid pattern."""
    positions = []
    for j in range(nr):
        for i in range(nc):
            if len(positions) >= N:
                break
            x = (i + 0.5) * w / nc
            y = (j + 0.5) * h / nr
            positions.append([x, y])
    while len(positions) < N:
        idx = len(positions)
        x = w / 2 + (idx - N / 2) * w / N * 0.5
        y = h / 2 + (idx - N / 2) * h / N * 0.5
        positions.append([x, y])
    return np.array(positions[:N])

def _slsqp_optimize(circles, w, h, max_iter=200):
    """
    SLSQP optimization: maximize sum of radii subject to non-overlap and boundary constraints.
    circles: (N, 3) array of [x, y, r]
    """
    x0 = circles.flatten()
    
    def objective(x):
        # Minimize negative sum of radii
        return -np.sum(x[2 * N:])
    
    def constraints(x):
        pos = x[:2 * N].reshape(N, 2)
        rad = x[2 * N:]
        cons = []
        
        # Boundary constraints
        for i in range(N):
            # x >= r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2 * N + i]})
            # x + r <= w
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: (w - x[i]) - x[2 * N + i]})
            # y >= r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N + i] - x[2 * N + i]})
            # y + r <= h
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: (h - x[N + i]) - x[2 * N + i]})
        
        # Non-overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                def cons_fn(x, i=i, j=j):
                    dx = x[i] - x[j]
                    dy = x[N + i] - x[N + j]
                    d = np.sqrt(dx * dx + dy * dy)
                    return d - (x[2 * N + i] + x[2 * N + j])
                cons.append({'type': 'ineq', 'fun': cons_fn})
        
        return cons
    
    max_r = min(w, h) / 2
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
    for i in range(N):
        bounds.append((1e-6, max_r))
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                         constraints=constraints,
                         options={'maxiter': max_iter, 'ftol': 1e-12})
        if result.success:
            return np.column_stack([result.x[:2 * N].reshape(N, 2), result.x[2 * N:]])
    except Exception:
        pass
    
    return circles

def _pack_for_aspect(w, h, deadline, max_force_iter=5000, max_slsqp_iter=100):
    """
    Pack circles in a rectangle of width w and height h.
    Uses force-directed layout followed by SLSQP refinement.
    """
    best_circles = None
    best_value = 0.0
    
    # Try multiple initial layouts
    layouts = [
        _hexagonal_init(w, h, 5, 4),
        _hexagonal_init(w, h, 4, 5),
        _grid_init(w, h, 5, 4),
        _grid_init(w, h, 4, 5),
        _grid_init(w, h, 3, 7),
        _grid_init(w, h, 7, 3),
    ]
    
    for init_pos in layouts:
        if time.time() > deadline - 10.0:
            break
        
        # Force-directed layout
        circles_fd = _force_directed_layout(w, h, init_pos, max_iter=max_force_iter)
        
        if _validate(circles_fd, w, h):
            value = _total_radius(circles_fd)
            if value > best_value:
                best_value = value
                best_circles = circles_fd
        
        # SLSQP refinement
        if time.time() < deadline - 5.0:
            circles_slsqp = _slsqp_optimize(circles_fd, w, h, max_iter=max_slsqp_iter)
            if _validate(circles_slsqp, w, h):
                value = _total_radius(circles_slsqp)
                if value > best_value:
                    best_value = value
                    best_circles = circles_slsqp
    
    return best_circles, best_value

def _nelder_mead_aspect_search(deadline):
    """
    Use Nelder-Mead to search for optimal aspect ratio (w/h).
    The objective is the negative of the best total radius found for that aspect ratio.
    """
    def objective(aspect):
        ar = aspect[0]
        if ar < 0.3 or ar > 3.0:
            return 100.0
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Quick force-directed layout with limited iterations
        init_pos = _hexagonal_init(w, h, 5, 4)
        circles = _force_directed_layout(w, h, init_pos, max_iter=500)
        
        if not _validate(circles, w, h):
            return 100.0
        
        return -_total_radius(circles)
    
    # Initial simplex: try a few aspect ratios
    x0 = np.array([1.0, 1.5, 0.7, 2.0, 0.5])
    
    try:
        result = minimize(objective, x0[:1], method='Nelder-Mead',
                         options={'maxiter': 50, 'xatol': 0.01, 'fatol': 1e-6})
        best_ar = result.x[0]
    except Exception:
        best_ar = 1.0
    
    return best_ar

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_value = 0.0
    best_result = None
    best_w = 1.0
    best_h = 1.0
    
    # Phase 1: Nelder-Mead aspect ratio search to find promising aspect ratios
    if time.time() < deadline - 60.0:
        best_ar = _nelder_mead_aspect_search(deadline)
    else:
        best_ar = 1.0
    
    # Phase 2: Detailed packing for multiple aspect ratios around the best found
    aspect_ratios_to_try = [best_ar, best_ar * 0.8, best_ar * 1.2, 1.0, 1.5, 0.75, 2.0/3, 0.5, 2.0]
    
    for ar in aspect_ratios_to_try:
        if time.time() > deadline - 15.0:
            break
        
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        circles, value = _pack_for_aspect(w, h, deadline, max_force_iter=3000, max_slsqp_iter=150)
        
        if circles is not None and value > best_value:
            best_value = value
            best_result = circles
            best_w = w
            best_h = h
    
    # Phase 3: Final SLSQP refinement on the best result
    if best_result is not None and time.time() < deadline - 5.0:
        refined = _slsqp_optimize(best_result, best_w, best_h, max_iter=200)
        if _validate(refined, best_w, best_h):
            value = _total_radius(refined)
            if value > best_value:
                best_value = value
                best_result = refined
    
    # Fallback: simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result
