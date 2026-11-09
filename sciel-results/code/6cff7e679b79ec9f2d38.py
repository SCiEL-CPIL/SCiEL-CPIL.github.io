"""Circle packing in perimeter-4 rectangle - structural rewrite with global NLP."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _validate(arr):
    """Validate a packing solution. Returns True if valid."""
    if arr is None or arr.shape != (N, 3):
        return False
    if not np.all(np.isfinite(arr)):
        return False
    
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    
    if np.any(r < -1e-9):
        return False
    
    # Check bounding box perimeter <= 4
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    w = x_max - x_min
    h = y_max - y_min
    if 2.0 * (w + h) > 4.0 + 1e-6:
        return False
    
    # Check non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.hypot(x[i] - x[j], y[i] - y[j])
            if dist < r[i] + r[j] - 1e-6:
                return False
    
    return True

def _compute_score(arr):
    """Compute sum of radii."""
    return np.sum(arr[:, 2])

def _grid_init(w, h, n_cols=None):
    """Generate a grid-based initial layout."""
    if n_cols is None:
        n_cols = max(1, int(np.round(np.sqrt(N * w / h))))
    n_rows = max(1, int(np.ceil(N / n_cols)))
    
    positions = np.zeros((N, 2))
    for i in range(N):
        row = i // n_cols
        col = i % n_cols
        cell_w = w / n_cols
        cell_h = h / n_rows
        x = (col + 0.5) * cell_w
        y = (row + 0.5) * cell_h
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)
        positions[i] = [x, y]
    
    min_dim = min(w / n_cols, h / n_rows)
    radii = np.full(N, min_dim * 0.35)
    return positions, radii

def _force_directed(positions, radii, w, h, n_iter=200, time_limit=10.0):
    """Force-directed relaxation to maximize radii."""
    start = time.time()
    positions = positions.copy()
    radii = radii.copy()
    
    for it in range(n_iter):
        if time.time() > start + time_limit:
            break
        
        # Update radii to max possible
        for i in range(N):
            max_r = min(positions[i, 0], w - positions[i, 0], 
                       positions[i, 1], h - positions[i, 1])
            for j in range(N):
                if i == j:
                    continue
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                avail = dist - radii[j]
                if avail < max_r:
                    max_r = avail
            radii[i] = max(0.0, max_r)
        
        # Compute forces
        forces = np.zeros_like(positions)
        for i in range(N):
            # Wall repulsion
            if positions[i, 0] - radii[i] < 0:
                forces[i, 0] += 2.0 * (positions[i, 0] - radii[i])
            if positions[i, 0] + radii[i] > w:
                forces[i, 0] -= 2.0 * (positions[i, 0] + radii[i] - w)
            if positions[i, 1] - radii[i] < 0:
                forces[i, 1] += 2.0 * (positions[i, 1] - radii[i])
            if positions[i, 1] + radii[i] > h:
                forces[i, 1] -= 2.0 * (positions[i, 1] + radii[i] - h)
        
        # Circle-circle repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[i] + radii[j]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    if dist > 1e-10:
                        overlap = min_dist - dist
                        fx = overlap * dx / dist
                        fy = overlap * dy / dist
                        forces[i, 0] -= fx
                        forces[i, 1] -= fy
                        forces[j, 0] += fx
                        forces[j, 1] += fy
                    else:
                        forces[i, 0] += np.random.randn() * 0.01
                        forces[i, 1] += np.random.randn() * 0.01
                        forces[j, 0] -= np.random.randn() * 0.01
                        forces[j, 1] -= np.random.randn() * 0.01
        
        # Apply forces with adaptive learning rate
        lr = 0.5 * (0.95 ** (it / 20.0))
        lr = max(0.01, lr)
        positions += forces * lr
        
        # Clamp to bounds
        for i in range(N):
            positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
            positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
    
    return positions, radii

def _slsqp_polish(positions, radii, w, h, time_limit=15.0):
    """SLSQP optimization to maximize sum of radii."""
    start = time.time()
    
    def objective(vars):
        if time.time() > start + time_limit:
            return 1e10
        pos = vars[:2*N].reshape(N, 2)
        rad = vars[2*N:]
        if np.any(rad < 0):
            return 1e10
        if np.any(pos[:, 0] < rad - 1e-9) or np.any(pos[:, 0] > w - rad + 1e-9):
            return 1e10
        if np.any(pos[:, 1] < rad - 1e-9) or np.any(pos[:, 1] > h - rad + 1e-9):
            return 1e10
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                if d < rad[i] + rad[j] - 1e-9:
                    return 1e10
        return -np.sum(rad)
    
    x0 = np.concatenate([positions.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((radii[i]*0.3, w - radii[i]*0.3))
        bounds.append((radii[i]*0.3, h - radii[i]*0.3))
        bounds.append((0.001, min(w, h)/2))
    
    try:
        res = minimize(objective, x0, method='SLSQP', 
                      bounds=bounds, options={'maxiter': 80, 'ftol': 1e-10})
        if res.fun < 1e9:
            pos_opt = res.x[:2*N].reshape(N, 2)
            rad_opt = res.x[2*N:]
            if np.all(rad_opt >= 0):
                return pos_opt, rad_opt
    except:
        pass
    
    return positions, radii

def _solve_fixed_rect(w, h, n_restarts=5, time_per_restart=12.0):
    """Solve for a fixed rectangle w x h."""
    best_result = None
    best_score = -1.0
    start_time = time.time()
    deadline = start_time + n_restarts * time_per_restart
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial layout with jitter
        positions, radii = _grid_init(w, h)
        if restart > 0:
            jitter = min(w, h) * 0.1
            positions += np.random.uniform(-jitter, jitter, size=positions.shape)
            positions[:, 0] = np.clip(positions[:, 0], 0, w)
            positions[:, 1] = np.clip(positions[:, 1], 0, h)
        
        # Force-directed relaxation
        positions_fd, radii_fd = _force_directed(positions, radii, w, h, 
                                                  n_iter=150, 
                                                  time_limit=time_per_restart * 0.5)
        
        # SLSQP polish
        positions_opt, radii_opt = _slsqp_polish(positions_fd, radii_fd, w, h,
                                                 time_limit=time_per_restart * 0.4)
        
        # Validate and score
        candidate = np.column_stack([positions_opt, radii_opt])
        if _validate(candidate):
            score = _compute_score(candidate)
            if score > best_score:
                best_score = score
                best_result = candidate.copy()
    
    # Also try force-directed result without SLSQP
    if best_result is None:
        positions, radii = _grid_init(w, h)
        positions_fd, radii_fd = _force_directed(positions, radii, w, h, n_iter=100, time_limit=5.0)
        candidate = np.column_stack([positions_fd, radii_fd])
        if _validate(candidate):
            best_score = _compute_score(candidate)
            best_result = candidate.copy()
    
    return best_result, best_score

def _solve_global_nlp(time_limit=30.0):
    """Global NLP: optimize positions, radii, AND rectangle dimensions simultaneously."""
    start_time = time.time()
    
    # Variables: x_i, y_i, r_i for i in 1..N, plus w, h
    # Total: 3*N + 2 variables
    # Constraints:
    #   2*(w + h) <= 4  -> w + h <= 2
    #   0 <= x_i <= w, 0 <= y_i <= h (with r_i margin)
    #   (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    #   r_i >= 0
    
    def objective(vars):
        if time.time() > start_time + time_limit:
            return 1e10
        pos = vars[:2*N].reshape(N, 2)
        rad = vars[2*N:3*N]
        w, h = vars[3*N], vars[3*N+1]
        
        if w <= 0 or h <= 0:
            return 1e10
        if w + h > 2.0 + 1e-9:
            return 1e10
        if np.any(rad < -1e-9):
            return 1e10
        # Check circles fit in rectangle
        for i in range(N):
            if pos[i, 0] < rad[i] - 1e-8 or pos[i, 0] > w - rad[i] + 1e-8:
                return 1e10
            if pos[i, 1] < rad[i] - 1e-8 or pos[i, 1] > h - rad[i] + 1e-8:
                return 1e10
        # Check non-overlap
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                if d < rad[i] + rad[j] - 1e-8:
                    return 1e10
        return -np.sum(rad)
    
    def constraints(vars):
        pos = vars[:2*N].reshape(N, 2)
        rad = vars[2*N:3*N]
        w, h = vars[3*N], vars[3*N+1]
        
        cons = []
        # Perimeter constraint: w + h <= 2
        cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - v[3*N] - v[3*N+1]})
        
        # Wall constraints
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i] - v[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N] - v[2*i] - v[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i+1] - v[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N+1] - v[2*i+1] - v[2*N+i]})
        
        # Non-overlap
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: 
                        (v[2*i] - v[2*j])**2 + (v[2*i+1] - v[2*j+1])**2 - 
                        (v[2*N+i] + v[2*N+j])**2
                })
        
        return cons
    
    # Initial guess: square-ish rectangle
    w0 = 1.0
    h0 = 1.0
    positions, radii = _grid_init(w0, h0)
    
    x0 = np.concatenate([positions.flatten(), radii, [w0, h0]])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0))  # x
        bounds.append((0.0, 2.0))  # y
        bounds.append((0.001, 1.0))  # radius
    bounds.append((0.1, 2.0))  # w
    bounds.append((0.1, 2.0))  # h
    
    try:
        res = minimize(objective, x0, method='SLSQP', 
                      bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10})
        if res.fun < 1e9:
            pos_opt = res.x[:2*N].reshape(N, 2)
            rad_opt = res.x[2*N:3*N]
            w_opt = res.x[3*N]
            h_opt = res.x[3*N+1]
            
            # Normalize: shift so min x = 0, min y = 0
            pos_opt[:, 0] -= np.min(pos_opt[:, 0])
            pos_opt[:, 1] -= np.min(pos_opt[:, 1])
            
            candidate = np.column_stack([pos_opt, rad_opt])
            if _validate(candidate):
                return candidate, _compute_score(candidate)
    except:
        pass
    
    return None, -1.0

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_result = None
    best_score = -1.0
    
    # Strategy 1: Try several fixed aspect ratios with multi-restart
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.0]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 40.0:
            break
        
        h = 2.0 / (ar + 1.0)
        w = ar * h
        
        # Use remaining time budget
        time_left = deadline - time.time() - 40.0
        n_restarts = max(2, int(time_left / 12.0))
        
        result, score = _solve_fixed_rect(w, h, n_restarts=n_restarts, 
                                          time_per_restart=12.0)
        
        if result is not None and score > best_score:
            best_score = score
            best_result = result.copy()
    
    # Strategy 2: Global NLP (optimize w, h, positions, radii jointly)
    if time.time() < deadline - 35.0:
        time_left = deadline - time.time() - 30.0
        result, score = _solve_global_nlp(time_limit=time_left)
        if result is not None and score > best_score:
            best_score = score
            best_result = result.copy()
    
    # Fallback: simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side * 0.85
            cells.append([x, y, r])
        best_result = np.array(cells)
        best_score = _compute_score(best_result)
    
    return best_result

# EVOLVE-BLOCK-END
