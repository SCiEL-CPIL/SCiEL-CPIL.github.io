import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _max_radius_for_circle(i, pos, radii, w, h):
    """Compute the maximum radius circle i can have given other circles' positions/radii."""
    # Wall constraints
    max_r = min(pos[i, 0], w - pos[i, 0], pos[i, 1], h - pos[i, 1])
    if max_r <= 0:
        return 0.0
    # Circle-circle constraints: r_i <= d - r_j for each j
    for j in range(N):
        if j == i:
            continue
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        d = np.sqrt(dx * dx + dy * dy)
        r_max = d - radii[j]
        if r_max < max_r:
            max_r = r_max
    return max(0.0, max_r)

def _refine_positions(pos, radii, w, h, max_iters=200, tol=1e-9):
    """
    Iteratively maximize each circle's radius by adjusting positions.
    Uses coordinate descent: for each circle, try to increase radius by moving it.
    """
    n = N
    best_score = np.sum(radii)
    
    for iteration in range(max_iters):
        improved = False
        for i in range(n):
            # Current max radius at current position
            r_max = _max_radius_for_circle(i, pos, radii, w, h)
            
            if r_max <= radii[i] + 1e-12:
                continue
            
            # Try to find a position where this circle can have a larger radius
            # Use local search: try small perturbations
            best_local_r = radii[i]
            best_local_pos = pos[i].copy()
            
            # Try moving in directions that might increase clearance
            # Heuristic: move toward the center of the largest empty space
            # Simpler: try random perturbations and gradient-like moves
            
            # Try moving to reduce the tightest constraint
            # Find which constraint is tightest
            tightness_wall = min(pos[i, 0], w - pos[i, 0], pos[i, 1], h - pos[i, 1])
            tightness_cc = None
            tight_j = -1
            for j in range(n):
                if j == i:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                clearance = d - radii[j] - radii[i]
                if tightness_cc is None or clearance < tightness_cc:
                    tightness_cc = clearance
                    tight_j = j
            
            # Direction to move: away from the tightest constraint
            if tight_j >= 0:
                dx = pos[i, 0] - pos[tight_j, 0]
                dy = pos[i, 1] - pos[tight_j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                if d > 1e-12:
                    dir_x = dx / d
                    dir_y = dy / d
                else:
                    dir_x, dir_y = 1.0, 0.0
            else:
                # Wall is tightest, move toward center
                dir_x = (w / 2 - pos[i, 0])
                dir_y = (h / 2 - pos[i, 1])
                d = np.sqrt(dir_x * dir_x + dir_y * dir_y)
                if d > 1e-12:
                    dir_x /= d
                    dir_y /= d
                else:
                    dir_x, dir_y = 1.0, 0.0
            
            # Try several step sizes in the direction
            for step_frac in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
                step = step_frac * (r_max - radii[i]) * 0.5
                new_x = pos[i, 0] + dir_x * step
                new_y = pos[i, 1] + dir_y * step
                
                # Keep within bounds
                new_x = np.clip(new_x, 0, w)
                new_y = np.clip(new_y, 0, h)
                
                # Check if this position allows larger radius
                old_pos = pos[i].copy()
                pos[i] = [new_x, new_y]
                r_test = _max_radius_for_circle(i, pos, radii, w, h)
                pos[i] = old_pos
                
                if r_test > best_local_r + 1e-10:
                    best_local_r = r_test
                    best_local_pos = np.array([new_x, new_y])
            
            # Also try random perturbations
            for _ in range(5):
                pert = np.random.randn(2) * 0.05 * min(w, h)
                new_x = np.clip(pos[i, 0] + pert[0], 0, w)
                new_y = np.clip(pos[i, 1] + pert[1], 0, h)
                
                old_pos = pos[i].copy()
                pos[i] = [new_x, new_y]
                r_test = _max_radius_for_circle(i, pos, radii, w, h)
                pos[i] = old_pos
                
                if r_test > best_local_r + 1e-10:
                    best_local_r = r_test
                    best_local_pos = np.array([new_x, new_y])
            
            if best_local_r > radii[i] + 1e-12:
                pos[i] = best_local_pos
                radii[i] = best_local_r
                improved = True
        
        if not improved:
            break
    
    return pos, radii, np.sum(radii)

def _sequential_place(pos, radii, w, h, start_index=0):
    """
    Sequentially place circles from start_index, finding best position for each.
    """
    for i in range(start_index, N):
        # Try to find a position for circle i that allows maximum radius
        best_r = 0.0
        best_pos = None
        
        # Try grid of candidate positions
        n_grid = 8
        for gx in range(n_grid):
            for gy in range(n_grid):
                cx = (gx + 0.5) * w / n_grid
                cy = (gy + 0.5) * h / n_grid
                
                # Check if this position is feasible (not overlapping existing circles)
                feasible = True
                for j in range(i):
                    dx = cx - pos[j, 0]
                    dy = cy - pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < radii[j] + 0.01:  # small margin
                        feasible = False
                        break
                
                if not feasible:
                    continue
                
                pos[i] = [cx, cy]
                r_max = _max_radius_for_circle(i, pos, radii, w, h)
                if r_max > best_r:
                    best_r = r_max
                    best_pos = np.array([cx, cy])
        
        # Also try some random positions
        for _ in range(20):
            cx = np.random.rand() * w
            cy = np.random.rand() * h
            
            feasible = True
            for j in range(i):
                dx = cx - pos[j, 0]
                dy = cy - pos[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                if d < radii[j] + 0.01:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            pos[i] = [cx, cy]
            r_max = _max_radius_for_circle(i, pos, radii, w, h)
            if r_max > best_r:
                best_r = r_max
                best_pos = np.array([cx, cy])
        
        if best_pos is not None and best_r > 0.001:
            pos[i] = best_pos
            radii[i] = best_r
        else:
            # Fallback: place at center with tiny radius
            pos[i] = [w / 2, h / 2]
            radii[i] = 0.001
    
    return pos, radii

def _pack_for_aspect(w, h, start_time, deadline, num_trials=5):
    """
    Pack circles in w x h rectangle using sequential placement + refinement.
    Returns best (pos, radii) and score.
    """
    best_score = 0
    best_pos = None
    best_radii = None
    
    for trial in range(num_trials):
        if time.time() > deadline - 2:
            break
        
        # Initialize with grid + jitter
        cols = int(np.ceil(np.sqrt(N * w / h)))
        rows = int(np.ceil(N / cols))
        if cols * rows < N:
            cols += 1
            rows = int(np.ceil(N / cols))
        
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        
        for i in range(N):
            col = i % cols
            row = i // cols
            pos[i, 0] = (col + 0.5) * w / cols
            pos[i, 1] = (row + 0.5) * h / rows
        
        # Add jitter for different trials
        if trial > 0:
            jitter = 0.1 * min(w, h) / cols
            pos[:, 0] += np.random.randn(N) * jitter
            pos[:, 1] += np.random.randn(N) * jitter
            pos[:, 0] = np.clip(pos[:, 0], 0, w)
            pos[:, 1] = np.clip(pos[:, 1], 0, h)
        
        # Sequential placement to set initial radii
        radii[:] = 0.0
        pos, radii = _sequential_place(pos, radii, w, h)
        
        # Refine
        pos, radii, score = _refine_positions(pos, radii, w, h, max_iters=100)
        
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_radii = radii.copy()
    
    return best_pos, best_radii, best_score

def _validate(pos, radii, w, h, tol=1e-7):
    """Check if configuration is valid (no overlaps, within bounds)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] < -tol or pos[i, 0] > w + tol:
            return False
        if pos[i, 1] < -tol or pos[i, 1] > h + tol:
            return False
        if radii[i] < -tol:
            return False
        # Wall check
        if radii[i] > pos[i, 0] + tol:
            return False
        if radii[i] > w - pos[i, 0] + tol:
            return False
        if radii[i] > pos[i, 1] + tol:
            return False
        if radii[i] > h - pos[i, 1] + tol:
            return False
    # Circle-circle check
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _fix_validity(pos, radii, w, h):
    """Shrink radii slightly to ensure strict validity."""
    for _ in range(50):
        if _validate(pos, radii, w, h, tol=1e-8):
            return pos, radii, True
        radii *= 0.99
    return pos, radii, False

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    np.random.seed(42)
    
    best_global_score = 0
    best_global_result = None
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 5:
            break
        
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        pos, radii, score = _pack_for_aspect(w, h, start_time, deadline, num_trials=3)
        
        if pos is None:
            continue
        
        # Validate and fix
        pos, radii, valid = _fix_validity(pos, radii, w, h)
        
        if valid and score > best_global_score:
            best_global_score = score
            best_global_result = np.column_stack([pos, radii])
    
    # Fallback if no valid result
    if best_global_result is None:
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        rad = np.zeros(N)
        r_val = 0.5 / side * 0.8
        for i in range(N):
            col = i % side
            row = i // side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
            rad[i] = r_val
        best_global_result = np.column_stack([pos, rad])
    
    return best_global_result
