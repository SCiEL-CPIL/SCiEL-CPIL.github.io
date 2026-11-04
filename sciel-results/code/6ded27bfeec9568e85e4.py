import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _soft_min(x, beta=50.0):
    """Smooth approximation of min(x) using log-sum-exp trick."""
    x = np.asarray(x)
    x_min = np.min(x)
    # For numerical stability, work with x - x_min >= 0
    shifted = x - x_min
    # soft_min ≈ x_min - (1/beta) * log(sum(exp(-beta * shifted)))
    # But since shifted >= 0, exp(-beta*shifted) <= 1, so no overflow
    # Underflow is fine (contributes ~0)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return x_min - log_sum / beta

def _compute_radii(pos, w, h):
    """Compute the maximum valid radius for each circle given positions."""
    n = len(pos)
    radii = np.zeros(n)
    for i in range(n):
        # Distance to walls
        wall_dists = np.array([
            pos[i, 0],
            w - pos[i, 0],
            pos[i, 1],
            h - pos[i, 1]
        ])
        min_dist = np.min(wall_dists)
        # Distance to other circles
        for j in range(n):
            if i != j:
                d = np.sqrt((pos[i, 0] - pos[j, 0])**2 + 
                           (pos[i, 1] - pos[j, 1])**2)
                min_dist = min(min_dist, d / 2.0)
        radii[i] = max(0.0, min_dist)
    return radii

def _objective_smooth(x, w, h, beta=50.0):
    """Smooth objective: maximize sum of soft_min radii."""
    n = N
    pos = x.reshape(n, 2)
    
    # Check bounds - return large penalty if out of bounds
    if np.any(pos[:, 0] < -0.01) or np.any(pos[:, 0] > w + 0.01) or \
       np.any(pos[:, 1] < -0.01) or np.any(pos[:, 1] > h + 0.01):
        return 1e6
    
    total = 0.0
    for i in range(n):
        # Wall distances
        wall_dists = np.array([
            pos[i, 0],
            w - pos[i, 0],
            pos[i, 1],
            h - pos[i, 1]
        ])
        # Circle distances
        circle_dists = []
        for j in range(n):
            if i != j:
                d = np.sqrt((pos[i, 0] - pos[j, 0])**2 + 
                           (pos[i, 1] - pos[j, 1])**2)
                circle_dists.append(d / 2.0)
        
        all_dists = np.concatenate([wall_dists, circle_dists])
        # Use soft_min for smoothness
        total += _soft_min(all_dists, beta)
    
    return -total

def _objective_grad(x, w, h, beta=50.0):
    """Objective with gradient computation."""
    n = N
    pos = x.reshape(n, 2)
    
    # Check bounds
    if np.any(pos[:, 0] < -0.01) or np.any(pos[:, 0] > w + 0.01) or \
       np.any(pos[:, 1] < -0.01) or np.any(pos[:, 1] > h + 0.01):
        return 1e6, np.zeros_like(x)
    
    grad = np.zeros_like(x)
    total = 0.0
    
    for i in range(n):
        # Wall distances and their gradients
        wall_dists = np.array([
            pos[i, 0],
            w - pos[i, 0],
            pos[i, 1],
            h - pos[i, 1]
        ])
        wall_grads = np.array([
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1]
        ])
        
        # Circle distances
        circle_dists = []
        circle_grads = []
        for j in range(n):
            if i != j:
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx**2 + dy**2)
                if d < 1e-12:
                    d = 1e-12
                circle_dists.append(d / 2.0)
                # Gradient of d/2 with respect to pos[i]
                circle_grads.append([dx / (2*d), dy / (2*d)])
        
        circle_dists = np.array(circle_dists) if circle_dists else np.array([])
        circle_grads = np.array(circle_grads) if circle_grads else np.zeros((0, 2))
        
        all_dists = np.concatenate([wall_dists, circle_dists])
        all_grads = np.concatenate([wall_grads, circle_grads], axis=0)
        
        # Soft min computation
        x_min = np.min(all_dists)
        shifted = all_dists - x_min
        exp_vals = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_vals)
        
        # soft_min = x_min - (1/beta) * log(sum_exp)
        total += x_min - np.log(sum_exp) / beta
        
        # Gradient of soft_min: -sum(exp_vals * grad_i) / (beta * sum_exp)
        # But we want gradient of -soft_min for minimization
        # d(soft_min)/d(pos) = -sum(exp_vals_i * grad_i) / (beta * sum_exp)
        # So d(-soft_min)/d(pos) = sum(exp_vals_i * grad_i) / (beta * sum_exp)
        
        weights = exp_vals / (beta * sum_exp)
        grad_contrib = np.sum(weights[:, np.newaxis] * all_grads, axis=0)
        grad[i * 2] += grad_contrib[0]
        grad[i * 2 + 1] += grad_contrib[1]
    
    return -total, grad

def _local_optimize(pos_init, w, h, max_iter=3000, tol=1e-10, beta=50.0):
    """Local optimization using gradient descent with adaptive step size."""
    pos = pos_init.copy()
    n = N
    x = pos.flatten().copy()
    
    # Adaptive step size
    step = 0.01
    best_obj = _objective_smooth(x, w, h, beta)
    best_x = x.copy()
    
    for iteration in range(max_iter):
        obj, grad = _objective_grad(x, w, h, beta)
        
        if not np.isfinite(obj):
            break
        
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()
        
        # Check convergence
        if np.linalg.norm(grad) < tol:
            break
        
        # Adaptive step size
        # Try the step
        x_trial = x - step * grad
        obj_trial = _objective_smooth(x_trial, w, h, beta)
        
        if obj_trial < obj:
            x = x_trial
            step *= 1.05  # Increase step slightly
        else:
            step *= 0.5
            if step < 1e-8:
                break
        
        # Keep positions in bounds (soft constraint)
        pos = x.reshape(n, 2)
        pos[:, 0] = np.clip(pos[:, 0], 0.001, w - 0.001)
        pos[:, 1] = np.clip(pos[:, 1], 0.001, h - 0.001)
        x = pos.flatten()
    
    return best_x.reshape(n, 2), best_obj

def _high_precision_polish(pos, w, h, max_iter=10000, tol=1e-12):
    """High-precision polish using very small steps and tight tolerance."""
    n = N
    x = pos.flatten().copy()
    
    step = 0.001
    best_obj = _objective_smooth(x, w, h, beta=100.0)
    best_x = x.copy()
    
    for iteration in range(max_iter):
        obj, grad = _objective_grad(x, w, h, beta=100.0)
        
        if not np.isfinite(obj):
            break
        
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()
        
        if np.linalg.norm(grad) < tol:
            break
        
        # Very conservative step
        x_trial = x - step * grad
        obj_trial = _objective_smooth(x_trial, w, h, beta=100.0)
        
        if obj_trial < obj:
            x = x_trial
            step *= 1.02
        else:
            step *= 0.7
            if step < 1e-10:
                break
        
        pos = x.reshape(n, 2)
        pos[:, 0] = np.clip(pos[:, 0], 0.0001, w - 0.0001)
        pos[:, 1] = np.clip(pos[:, 1], 0.0001, h - 0.0001)
        x = pos.flatten()
    
    return best_x.reshape(n, 2)

def _grid_initialization(w, h, n_circles=N):
    """Initialize circles in a grid pattern."""
    cols = int(np.ceil(np.sqrt(n_circles * w / h)))
    rows = int(np.ceil(n_circles / cols))
    
    positions = np.zeros((n_circles, 2))
    for i in range(n_circles):
        col = i % cols
        row = i // cols
        positions[i, 0] = (col + 0.5) * w / cols
        positions[i, 1] = (row + 0.5) * h / rows
    
    return positions

def _hexagonal_initialization(w, h, n_circles=N):
    """Initialize circles in a hexagonal pattern."""
    positions = []
    # Estimate spacing
    area = w * h
    avg_spacing = np.sqrt(area / n_circles)
    
    row_height = avg_spacing * np.sqrt(3) / 2
    cols = int(np.ceil(w / avg_spacing))
    
    row = 0
    while len(positions) < n_circles and row < 50:
        offset = (row % 2) * avg_spacing / 2
        col = 0
        while len(positions) < n_circles and col < 50:
            x = offset + col * avg_spacing
            y = row * row_height
            if 0 < x < w and 0 < y < h:
                positions.append([x, y])
            col += 1
        row += 1
    
    positions = np.array(positions[:n_circles])
    if len(positions) < n_circles:
        # Fill remaining with grid
        remaining = n_circles - len(positions)
        grid_pos = _grid_initialization(w, h, remaining)
        positions = np.vstack([positions, grid_pos])
    
    return positions

def _random_initialization(w, h, n_circles=N, seed=42):
    """Random initialization."""
    rng = np.random.default_rng(seed)
    positions = np.random.uniform(0.05, [w - 0.05, h - 0.05], size=(n_circles, 2), generator=rng)
    return positions

def _pack_for_aspect(w, h, n_circles=N):
    """Pack circles in w x h rectangle, return (positions, radii) maximizing sum of radii."""
    # Try multiple initializations
    inits = [
        _grid_initialization(w, h, n_circles),
        _hexagonal_initialization(w, h, n_circles),
        _random_initialization(w, h, n_circles, seed=1),
        _random_initialization(w, h, n_circles, seed=2),
    ]
    
    best_pos = None
    best_score = -np.inf
    
    for init in inits:
        # Local optimization
        pos, obj = _local_optimize(init, w, h, max_iter=2000, tol=1e-8, beta=30.0)
        
        # High precision polish
        pos = _high_precision_polish(pos, w, h, max_iter=3000, tol=1e-10)
        
        # Compute actual radii
        radii = _compute_radii(pos, w, h)
        score = np.sum(radii)
        
        if score > best_score:
            best_score = score
            best_pos = pos
    
    if best_pos is None:
        best_pos = _grid_initialization(w, h, n_circles)
    
    radii = _compute_radii(best_pos, w, h)
    return best_pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_score = 0
    best_result = None
    
    # Try various aspect ratios
    # Focus on promising ranges based on the problem structure
    aspect_ratios = np.concatenate([
        np.linspace(0.5, 1.0, 10),
        np.linspace(1.0, 2.0, 15),
        np.linspace(2.0, 3.0, 8)
    ])
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        pos, radii = _pack_for_aspect(w, h)
        score = np.sum(radii)
        
        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos, radii])
    
    # Fallback to seed if nothing works
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
