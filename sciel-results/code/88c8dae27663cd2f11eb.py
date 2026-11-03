import numpy as np
import time

N = 26

def _max_radius_for_circle(i, centers, radii):
    """Compute the maximum radius circle i can have given other circles' positions."""
    x, y = centers[i]
    max_r = min(x, 1 - x, y, 1 - y)
    for j in range(N):
        if j == i:
            continue
        dx = x - centers[j][0]
        dy = y - centers[j][1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > 0:
            max_r = min(max_r, dist - radii[j])
    return max(0.0, max_r)

def _eval_packing(centers, radii):
    """Evaluate a packing: return (sum_of_radii, is_valid)."""
    sum_r = np.sum(radii)
    
    # Check boundaries
    for i in range(N):
        if centers[i][0] < -1e-9 or centers[i][0] > 1 + 1e-9:
            return sum_r, False
        if centers[i][1] < -1e-9 or centers[i][1] > 1 + 1e-9:
            return sum_r, False
        if radii[i] < -1e-9:
            return sum_r, False
        if radii[i] > centers[i][0] + 1e-9:
            return sum_r, False
        if radii[i] > 1 - centers[i][0] + 1e-9:
            return sum_r, False
        if radii[i] > centers[i][1] + 1e-9:
            return sum_r, False
        if radii[i] > 1 - centers[i][1] + 1e-9:
            return sum_r, False
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-9:
                return sum_r, False
    
    return sum_r, True

def _best_fit_local_search(centers, radii, max_iters=2000, deadline=None):
    """Iteratively move each circle to maximize its radius given others fixed."""
    centers = centers.copy()
    radii = radii.copy()
    current_score = np.sum(radii)
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            # Try to maximize radius of circle i by moving it
            best_pos = centers[i]
            best_r = radii[i]
            
            # Sample candidate positions
            candidates = []
            
            # Current position
            candidates.append((centers[i][0], centers[i][1]))
            
            # Random perturbations
            for _ in range(20):
                angle = np.random.uniform(0, 2 * np.pi)
                dist = np.random.uniform(0, 0.1)
                nx = centers[i][0] + dist * np.cos(angle)
                ny = centers[i][1] + dist * np.sin(angle)
                if 0 <= nx <= 1 and 0 <= ny <= 1:
                    candidates.append((nx, ny))
            
            # Try each candidate
            for cx, cy in candidates:
                # Compute max radius at this position
                max_r = min(cx, 1 - cx, cy, 1 - cy)
                for j in range(N):
                    if j == i:
                        continue
                    dx = cx - centers[j][0]
                    dy = cy - centers[j][1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist > 0:
                        max_r = min(max_r, dist - radii[j])
                max_r = max(0.0, max_r)
                
                if max_r > best_r + 1e-10:
                    best_r = max_r
                    best_pos = (cx, cy)
                    improved = True
        
        # Update if improved
        if improved:
            centers[i] = best_pos
            radii[i] = best_r
    
    return centers, radii, np.sum(radii)

def _grid_init():
    """Create initial grid packing."""
    # Use a 5x5 grid + 1 extra
    centers = []
    radii = []
    for i in range(5):
        for j in range(5):
            centers.append(((i + 0.5) / 5.0, (j + 0.5) / 5.0))
            radii.append(0.09)
    # Add one more
    centers.append((0.5, 0.5))
    radii.append(0.09)
    
    return np.array(centers), np.array(radii)

def _random_init(seed=42):
    """Create random initial packing."""
    np.random.seed(seed)
    centers = np.random.rand(N, 2)
    radii = np.full(N, 0.05)
    return centers, radii

def _hex_init():
    """Create hexagonal-like initial packing."""
    centers = []
    radii = []
    # Hexagonal packing
    r0 = 0.09
    dy = r0 * 2 * 0.866  # sqrt(3)/2
    for row in range(6):
        x_offset = 0 if row % 2 == 0 else r0
        for col in range(5):
            x = r0 + col * 2 * r0 + x_offset
            y = r0 + row * dy
            if x < 1 and y < 1:
                centers.append((x, y))
                radii.append(r0)
            if len(centers) >= N:
                break
        if len(centers) >= N:
            break
    
    # If we don't have enough, add random
    while len(centers) < N:
        centers.append((np.random.rand(), np.random.rand()))
        radii.append(0.05)
    
    return np.array(centers[:N]), np.array(radii[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = -np.inf
    best_centers = None
    best_radii = None
    
    # Generate multiple initial configurations
    inits = []
    
    # Grid init
    c, r = _grid_init()
    inits.append((c, r))
    
    # Random inits
    for seed in [42, 123, 456, 789, 1011, 2020]:
        c, r = _random_init(seed)
        inits.append((c, r))
    
    # Hex init
    c, r = _hex_init()
    inits.append((c, r))
    
    # Run best-fit local search on each
    for centers, radii in inits:
        if time.time() > deadline:
            break
        
        try:
            # First, compute proper radii for initial positions
            for i in range(N):
                radii[i] = _max_radius_for_circle(i, centers, radii)
            
            # Run local search
            c_opt, r_opt, score = _best_fit_local_search(centers, radii, 
                                                         max_iters=500, 
                                                         deadline=deadline)
            
            # Validate
            valid_score, is_valid = _eval_packing(c_opt, r_opt)
            if is_valid and valid_score > best_score:
                best_score = valid_score
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
        except Exception:
            pass
    
    # If no valid solution found, use a simple grid fallback
    if best_centers is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells)
    
    return np.column_stack((best_centers, best_radii))
