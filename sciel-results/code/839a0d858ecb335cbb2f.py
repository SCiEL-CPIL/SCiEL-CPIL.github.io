"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _perimeter_constraint(w, h):
    """Perimeter constraint: 2*(w+h) <= 4, i.e. w+h <= 2."""
    return 2.0 - (w + h)


def _objective_and_grad(params):
    """
    params: [w, h, x_1..x_N, y_1..y_N, r_1..r_N]
    Objective: maximize sum(r_i) -> minimize -sum(r_i)
    
    We use a smooth penalty for constraints to allow gradient-based optimization.
    """
    w = params[0]
    h = params[1]
    xy = params[2:2+2*N].reshape(N, 2)
    r = params[2+2*N:2+3*N]
    
    # Perimeter penalty (soft)
    perim_viol = max(0.0, 2*(w+h) - 4.0)
    
    # Boundary violations
    boundary_viol = 0.0
    for i in range(N):
        x, y = xy[i]
        ri = r[i]
        # Left: x - r >= 0
        boundary_viol += max(0.0, ri - x)**2
        # Right: x + r <= w
        boundary_viol += max(0.0, x + ri - w)**2
        # Bottom: y - r >= 0
        boundary_viol += max(0.0, ri - y)**2
        # Top: y + r <= h
        boundary_viol += max(0.0, y + ri - h)**2
    
    # Overlap violations
    overlap_viol = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = xy[i,0] - xy[j,0]
            dy = xy[i,1] - xy[j,1]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap_viol += (min_dist - dist)**2
    
    penalty = 1000.0 * (perim_viol**2 + boundary_viol + overlap_viol)
    obj = -np.sum(r) + penalty
    
    # Gradient (numerical)
    grad = np.zeros_like(params)
    eps = 1e-7
    for k in range(len(params)):
        orig = params[k]
        params[k] = orig + eps
        obj_plus = _objective_and_grad_val(params)
        params[k] = orig
        grad[k] = (obj_plus - obj) / eps
        params[k] = orig  # restore
    
    return obj, grad


def _objective_and_grad_val(params):
    """Just the value, for gradient computation."""
    w = params[0]
    h = params[1]
    xy = params[2:2+2*N].reshape(N, 2)
    r = params[2+2*N:2+3*N]
    
    perim_viol = max(0.0, 2*(w+h) - 4.0)
    
    boundary_viol = 0.0
    for i in range(N):
        x, y = xy[i]
        ri = r[i]
        boundary_viol += max(0.0, ri - x)**2
        boundary_viol += max(0.0, x + ri - w)**2
        boundary_viol += max(0.0, ri - y)**2
        boundary_viol += max(0.0, y + ri - h)**2
    
    overlap_viol = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = xy[i,0] - xy[j,0]
            dy = xy[i,1] - xy[j,1]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap_viol += (min_dist - dist)**2
    
    penalty = 1000.0 * (perim_viol**2 + boundary_viol + overlap_viol)
    return -np.sum(r) + penalty


def _is_valid(w, h, xy, r, tol=1e-6):
    """Check if the configuration is valid."""
    if w <= 0 or h <= 0:
        return False
    if 2*(w+h) > 4.0 + tol:
        return False
    for i in range(N):
        x, y = xy[i]
        ri = r[i]
        if ri < -tol:
            return False
        if x - ri < -tol or x + ri > w + tol or y - ri < -tol or y + ri > h + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = xy[i,0] - xy[j,0]
            dy = xy[i,1] - xy[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _score(w, h, xy, r):
    """Compute the raw value (sum of radii) if valid, else -inf."""
    if not _is_valid(w, h, xy, r):
        return -np.inf
    return np.sum(r)


def _initial_guess(w, h):
    """Create an initial grid-based guess."""
    # Use a grid of N circles
    cols = int(np.ceil(np.sqrt(N * w / h))) if h > 0 else int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    
    cw = w / cols
    ch = h / rows
    
    positions = []
    radii = []
    for i in range(cols):
        for j in range(rows):
            if len(positions) >= N:
                break
            cx = (i + 0.5) * cw
            cy = (j + 0.5) * ch
            cr = min(cw, ch) * 0.45
            positions.append([cx, cy])
            radii.append(cr)
    
    while len(positions) < N:
        positions.append([w/2, h/2])
        radii.append(0.01)
    
    return np.array(positions[:N]), np.array(radii[:N])


def _refine(w, h, xy, r, time_limit=10.0):
    """
    Refine a packing configuration using iterative constraint projection.
    Returns improved (xy, r) or the original if no improvement.
    """
    start_time = time.time()
    best_score = _score(w, h, xy, r)
    if best_score == -np.inf:
        # Try to fix validity first
        xy, r = _fix_validity(w, h, xy, r)
        best_score = _score(w, h, xy, r)
    
    xy = xy.copy()
    r = r.copy()
    
    max_iter = 5000
    lr = 0.01
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        force = np.zeros_like(xy)
        
        # Boundary forces
        for i in range(N):
            x, y = xy[i]
            ri = r[i]
            # Left wall
            if x < ri:
                force[i, 0] += (ri - x) * 10
            # Right wall
            if x > w - ri:
                force[i, 0] -= (x - (w - ri)) * 10
            # Bottom wall
            if y < ri:
                force[i, 1] += (ri - y) * 10
            # Top wall
            if y > h - ri:
                force[i, 1] -= (y - (h - ri)) * 10
        
        # Overlap forces
        for i in range(N):
            for j in range(i+1, N):
                dx = xy[i,0] - xy[j,0]
                dy = xy[i,1] - xy[j,1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + 1e-12)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = (min_dist - dist)
                    if dist > 1e-10:
                        fx = dx / dist * overlap * 5
                        fy = dy / dist * overlap * 5
                    else:
                        # Random push
                        angle = np.random.uniform(0, 2*np.pi)
                        fx = np.cos(angle) * overlap * 5
                        fy = np.sin(angle) * overlap * 5
                    force[i, 0] += fx
                    force[i, 1] += fy
                    force[j, 0] -= fx
                    force[j, 1] -= fy
        
        # Update positions
        xy += lr * force
        
        # Clamp to boundary
        for i in range(N):
            xy[i, 0] = np.clip(xy[i, 0], r[i], w - r[i])
            xy[i, 1] = np.clip(xy[i, 1], r[i], h - r[i])
        
        # Try to increase radii slightly
        new_r = r * (1.0 + 0.001)
        # Check if still valid with slightly larger radii
        if _is_valid(w, h, xy, new_r, tol=1e-5):
            r = new_r
        else:
            # Try smaller increase
            new_r = r * (1.0 + 0.0001)
            if _is_valid(w, h, xy, new_r, tol=1e-5):
                r = new_r
        
        # Check improvement
        current_score = _score(w, h, xy, r)
        if current_score > best_score:
            best_score = current_score
    
    return xy, r


def _fix_validity(w, h, xy, r):
    """Fix a configuration to be valid by shrinking radii and adjusting positions."""
    # Shrink radii until no overlaps
    for _ in range(100):
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                dx = xy[i,0] - xy[j,0]
                dy = xy[i,1] - xy[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j]:
                    valid = False
                    # Shrink both
                    r[i] *= 0.95
                    r[j] *= 0.95
        if valid:
            break
    
    # Clamp to boundaries
    for i in range(N):
        xy[i, 0] = np.clip(xy[i, 0], r[i], max(r[i], w - r[i]))
        xy[i, 1] = np.clip(xy[i, 1], r[i], max(r[i], h - r[i]))
        # Ensure radius fits
        r[i] = min(r[i], min(xy[i,0], w - xy[i,0], xy[i,1], h - xy[i,1]))
        r[i] = max(r[i], 0.001)
    
    return xy, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_score = -np.inf
    best_result = None
    
    # Try several aspect ratios
    aspect_ratios = [1.0, 0.5, 2.0, 0.75, 4/3, 1.5, 0.6, 5/3, 0.8, 1.25]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # w + h = 2 (perimeter = 4)
        # h = w / ar
        # w + w/ar = 2 -> w(1 + 1/ar) = 2 -> w = 2*ar/(ar+1)
        w = 2.0 * ar / (ar + 1.0)
        h = 2.0 / (ar + 1.0)
        
        # Initial guess
        xy, r = _initial_guess(w, h)
        
        # Refine
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
        time_budget = min(time_left * 0.8, 15.0)
        xy, r = _refine(w, h, xy, r, time_limit=time_budget)
        
        score = _score(w, h, xy, r)
        if score > best_score:
            best_score = score
            best_result = (w, h, xy.copy(), r.copy())
    
    # If we have time, try to improve the best result further
    if best_result is not None and time.time() < deadline - 5.0:
        w, h, xy, r = best_result
        time_left = deadline - time.time() - 5.0
        if time_left > 2.0:
            xy, r = _refine(w, h, xy, r, time_limit=min(time_left, 30.0))
            score = _score(w, h, xy, r)
            if score > best_score:
                best_score = score
                best_result = (w, h, xy.copy(), r.copy())
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    w, h, xy, r = best_result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = xy[i, 0]
        result[i, 1] = xy[i, 1]
        result[i, 2] = r[i]
    
    return result
