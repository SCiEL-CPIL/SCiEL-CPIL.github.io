# EVOLVE-BLOCK-START
import numpy as np
import time


def _is_inside_base(u, v):
    """Check if point (u, v) is in the base L-shaped corridor."""
    return ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
           ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))


def _compute_sofa_area(path, deadline=None):
    """
    Compute the area of the sofa (intersection of all corridor placements)
    for a given path. The evaluator adds (0,0,0) before and (-2,0,90) after.
    """
    placements = np.vstack([
        np.array([[0.0, 0.0, 0.0]]),
        path,
        np.array([[-2.0, 0.0, 90.0]])
    ])
    
    # Densify
    densified = [placements[0]]
    for i in range(len(placements) - 1):
        p1 = placements[i]
        p2 = placements[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        if m > 1:
            for j in range(1, m):
                t = j / m
                interp = p1 + t * (p2 - p1)
                densified.append(interp)
        densified.append(p2)
    
    densified = np.array(densified)
    
    # Sofa grid points
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    sofa_points = np.column_stack([X.ravel(), Y.ravel()])
    n_points = len(sofa_points)
    
    inside_all = np.ones(n_points, dtype=bool)
    
    for k in range(len(densified)):
        x, y, a = densified[k]
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(a):
            continue
        a_rad = np.radians(a)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        dx = sofa_points[:, 0] - x
        dy = sofa_points[:, 1] - y
        
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        inside = _is_inside_base(u, v)
        inside_all &= inside
        
        if not inside_all.any():
            break
        
        if deadline is not None and time.time() > deadline:
            break
    
    count = inside_all.sum()
    return count * 0.0001


def _soft_sofa_area(path, temperature=0.1, deadline=None):
    """
    Soft surrogate: for each grid point, compute a smooth measure of how well
    it fits inside all placements. Use log-sum-exp of negative margins.
    Returns a smooth area estimate.
    """
    placements = np.vstack([
        np.array([[0.0, 0.0, 0.0]]),
        path,
        np.array([[-2.0, 0.0, 90.0]])
    ])
    
    densified = [placements[0]]
    for i in range(len(placements) - 1):
        p1 = placements[i]
        p2 = placements[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        if m > 1:
            for j in range(1, m):
                t = j / m
                interp = p1 + t * (p2 - p1)
                densified.append(interp)
        densified.append(p2)
    
    densified = np.array(densified)
    
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    sofa_points = np.column_stack([X.ravel(), Y.ravel()])
    n_points = len(sofa_points)
    
    # For each point, compute the minimum "margin" across all placements
    # Margin = distance to the nearest boundary (positive if inside)
    # Use a simpler approach: count how many placements contain the point,
    # then use a smooth step function
    inside_counts = np.zeros(n_points, dtype=np.float64)
    
    for k in range(len(densified)):
        x, y, a = densified[k]
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(a):
            continue
        a_rad = np.radians(a)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        dx = sofa_points[:, 0] - x
        dy = sofa_points[:, 1] - y
        
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        inside = _is_inside_base(u, v).astype(np.float64)
        inside_counts += inside
        
        if deadline is not None and time.time() > deadline:
            break
    
    n_placements = len(densified)
    # Fraction of placements containing each point
    fractions = inside_counts / n_placements
    # Soft area: use sigmoid to smoothly count points that are "mostly" inside
    # A point contributes sigmoid((fraction - 0.5) / temperature)
    soft_counts = 1.0 / (1.0 + np.exp(-(fractions - 0.5) / temperature))
    
    return soft_counts.sum() * 0.0001


def moving_sofa_2d() -> np.ndarray:
    """
    Multi-phase optimization: soft surrogate for coarse search, exact for refinement.
    """
    start_time = time.time()
    deadline = start_time + 570.0
    
    # Good initial path
    best_path = np.array([
        [-0.05, 0.05, 10.0],
        [-0.15, 0.15, 25.0],
        [-0.25, 0.25, 40.0],
        [-0.35, 0.25, 55.0],
        [-0.45, 0.15, 65.0],
        [-0.55, 0.15, 75.0],
        [-0.65, 0.25, 85.0],
    ], dtype=np.float64)
    
    # Phase 1: Coarse search with soft surrogate
    phase1_deadline = start_time + 200.0
    best_soft = _soft_sofa_area(best_path, temperature=0.1, deadline=phase1_deadline)
    step_size = 0.1
    n_points = len(best_path)
    
    for step in range(300):
        if time.time() > phase1_deadline - 5.0:
            break
        
        perturbation = np.random.randn(n_points, 3) * step_size
        candidate = best_path + perturbation
        
        # Enforce monotonic angles
        for i in range(1, n_points):
            if candidate[i, 2] < candidate[i-1, 2] + 0.5:
                candidate[i, 2] = candidate[i-1, 2] + 0.5
            candidate[i, 2] = min(candidate[i, 2], 89.0)
        candidate[0, 2] = max(candidate[0, 2], 5.0)
        
        # Bounds
        candidate[:, 0] = np.clip(candidate[:, 0], -1.5, 0.5)
        candidate[:, 1] = np.clip(candidate[:, 1], -0.5, 0.5)
        
        soft = _soft_sofa_area(candidate, temperature=0.1, deadline=phase1_deadline)
        
        if soft > best_soft:
            best_path = candidate
            best_soft = soft
            step_size = min(step_size * 1.1, 0.2)
        else:
            step_size = max(step_size * 0.9, 0.02)
    
    # Phase 2: Exact evaluation refinement
    best_area = _compute_sofa_area(best_path, deadline)
    step_size = 0.05
    
    for step in range(200):
        if time.time() > deadline - 30.0:
            break
        
        perturbation = np.random.randn(n_points, 3) * step_size
        candidate = best_path + perturbation
        
        for i in range(1, n_points):
            if candidate[i, 2] < candidate[i-1, 2] + 0.5:
                candidate[i, 2] = candidate[i-1, 2] + 0.5
            candidate[i, 2] = min(candidate[i, 2], 89.0)
        candidate[0, 2] = max(candidate[0, 2], 5.0)
        
        candidate[:, 0] = np.clip(candidate[:, 0], -1.5, 0.5)
        candidate[:, 1] = np.clip(candidate[:, 1], -0.5, 0.5)
        
        area = _compute_sofa_area(candidate, deadline)
        
        if area > best_area + 0.0001:
            best_path = candidate
            best_area = area
            step_size = min(step_size * 1.1, 0.15)
        else:
            step_size = max(step_size * 0.88, 0.01)
    
    # Phase 3: Fine polish
    step_size = 0.01
    for step in range(100):
        if time.time() > deadline - 10.0:
            break
        
        perturbation = np.random.randn(n_points, 3) * step_size
        candidate = best_path + perturbation
        
        for i in range(1, n_points):
            if candidate[i, 2] < candidate[i-1, 2] + 0.5:
                candidate[i, 2] = candidate[i-1, 2] + 0.5
            candidate[i, 2] = min(candidate[i, 2], 89.0)
        candidate[0, 2] = max(candidate[0, 2], 5.0)
        
        candidate[:, 0] = np.clip(candidate[:, 0], -1.5, 0.5)
        candidate[:, 1] = np.clip(candidate[:, 1], -0.5, 0.5)
        
        area = _compute_sofa_area(candidate, deadline)
        
        if area > best_area + 0.00005:
            best_path = candidate
            best_area = area
        else:
            step_size = max(step_size * 0.9, 0.003)
    
    return best_path


# EVOLVE-BLOCK-END
