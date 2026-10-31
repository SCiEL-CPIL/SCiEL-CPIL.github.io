"""Circle packing in perimeter-4 rectangle: iterative relaxation with multi-start."""
import numpy as np
import time
import random

N = 21
START = time.time()
DEADLINE = START + 145.0  # Safety margin before 150s kill

def _time_ok():
    return time.time() < DEADLINE - 5.0

def _validate(circles):
    """Check disjointness and perimeter constraint. Returns (is_valid, sum_radii)."""
    if circles is None or len(circles) != N:
        return False, 0.0
    if np.any(circles[:, 2] <= 0):
        return False, 0.0
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # Bounding box
    x_min = np.min(xs - rs)
    x_max = np.max(xs + rs)
    y_min = np.min(ys - rs)
    y_max = np.max(ys + rs)
    width = x_max - x_min
    height = y_max - y_min
    perimeter = 2.0 * (width + height)
    if perimeter > 4.0 + 1e-9:
        return False, 0.0
    
    # Disjointness
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return False, 0.0
    return True, float(np.sum(rs))

def _relax(cx, cy, r, W, H, n_iter=10):
    """
    Iteratively increase each circle's radius to the maximum possible 
    given its position and the other circles.
    """
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    for _ in range(n_iter):
        improved = False
        for i in range(N):
            # Max radius from boundaries
            r_bound = min(cx[i], W - cx[i], cy[i], H - cy[i])
            
            # Max radius from other circles
            r_max = r_bound
            for j in range(N):
                if i == j:
                    continue
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx * dx + dy * dy)
                # Circle i can be at most dist - r[j]
                r_candidate = dist - r[j]
                if r_candidate < r_max:
                    r_max = r_candidate
            
            r_max = max(r_max, 1e-6)
            if r_max > r[i] + 1e-10:
                r[i] = r_max
                improved = True
        
        if not improved:
            break
    
    return cx, cy, r

def _optimize_positions(cx, cy, r, W, H, step=0.02, n_iter=20):
    """
    Try to move circles to positions that allow for larger radii.
    This is a local search step.
    """
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    for _ in range(n_iter):
        for i in range(N):
            best_cx, best_cy = cx[i], cy[i]
            best_r = r[i]
            
            # Try 8 neighbors + center
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    new_cx = cx[i] + dx
                    new_cy = cy[i] + dy
                    
                    # Check bounds
                    if new_cx < 0 or new_cx > W or new_cy < 0 or new_cy > H:
                        continue
                    
                    # Calculate max radius at this new position
                    r_bound = min(new_cx, W - new_cx, new_cy, H - new_cy)
                    r_max = r_bound
                    for j in range(N):
                        if i == j:
                            continue
                        dx2 = new_cx - cx[j]
                        dy2 = new_cy - cy[j]
                        dist = np.sqrt(dx2 * dx2 + dy2 * dy2)
                        r_candidate = dist - r[j]
                        if r_candidate < r_max:
                            r_max = r_candidate
                    
                    r_max = max(r_max, 1e-6)
                    
                    if r_max > best_r + 1e-10:
                        best_r = r_max
                        best_cx = new_cx
                        best_cy = new_cy
            
            cx[i] = best_cx
            cy[i] = best_cy
            r[i] = best_r
            
    return cx, cy, r

def _make_grid(kx, ky, W, H):
    """Create a kx by ky grid of circles, take first N."""
    pts = []
    for j in range(ky):
        for i in range(kx):
            cx = (i + 0.5) * W / kx
            cy = (j + 0.5) * H / ky
            r = min(cx, cy, W - cx, H - cy) * 0.9
            r = max(r, 1e-4)
            pts.append([cx, cy, r])
    
    arr = np.array(pts)
    if len(arr) < N:
        for _ in range(N - len(arr)):
            arr = np.vstack([arr, [W/2, H/2, 1e-4]])
    return arr[:N]

def _hexagonal_layout(W, H):
    """Create a hexagonal packing layout."""
    pts = []
    # Estimate row height for hex packing of unit circles
    # d = 2r, row_h = r * sqrt(3)
    # Let's estimate a base radius
    base_r = min(W, H) / 5.0
    
    row_h = base_r * np.sqrt(3)
    n_rows = int(H / row_h) + 2
    n_cols = int(W / (2 * base_r)) + 2
    
    for j in range(n_rows):
        y = j * row_h
        if y > H:
            break
        offset = (j % 2) * base_r
        for i in range(n_cols):
            x = i * 2 * base_r + offset
            if x > W:
                break
            
            # Check if position fits
            r_max = min(x, W - x, y, H - y)
            if r_max > base_r * 0.5:
                r = base_r * 0.9
                r = min(r, r_max)
                if r > 1e-4:
                    pts.append([x, y, r])
    
    if len(pts) == 0:
        return _make_grid(5, 5, W, H)
    
    arr = np.array(pts)
    # Sort by radius descending to keep largest circles
    order = np.argsort(-arr[:, 2])
    arr = arr[order]
    
    if len(arr) < N:
        for _ in range(N - len(arr)):
            arr = np.vstack([arr, [W/2, H/2, 1e-4]])
    
    return arr[:N]

def _random_layout(W, H):
    """Create a random layout."""
    pts = []
    for _ in range(N):
        cx = random.uniform(0, W)
        cy = random.uniform(0, H)
        r = 0.05
        pts.append([cx, cy, r])
    return np.array(pts)

def _optimize_one(W, H, x0, time_budget):
    """Optimize a single layout for given W, H."""
    cx = x0[:, 0].copy()
    cy = x0[:, 1].copy()
    r = x0[:, 2].copy()
    
    # Ensure initial validity by clamping
    cx = np.clip(cx, 0, W)
    cy = np.clip(cy, 0, H)
    r = np.maximum(r, 1e-6)
    for i in range(N):
        r_bound = min(cx[i], W - cx[i], cy[i], H - cy[i])
        r[i] = min(r[i], r_bound)
    
    start_time = time.time()
    best_score = 0.0
    best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
    
    # Main optimization loop
    while time.time() - start_time < time_budget and _time_ok():
        # 1. Relax radii
        cx, cy, r = _relax(cx, cy, r, W, H, n_iter=5)
        
        # 2. Optimize positions
        cx, cy, r = _optimize_positions(cx, cy, r, W, H, step=0.02, n_iter=5)
        
        # 3. Relax radii again
        cx, cy, r = _relax(cx, cy, r, W, H, n_iter=5)
        
        # Check score
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_cx, best_cy, best_r = cx.copy(), cy.copy(), r.copy()
            
        # Periodic restart with perturbation if stuck
        if random.random() < 0.05:
            cx, cy, r = _relax(best_cx + 0.01 * np.random.randn(N), 
                               best_cy + 0.01 * np.random.randn(N), 
                               best_r, W, H, n_iter=1)
            # Clamp to bounds
            cx = np.clip(cx, 0, W)
            cy = np.clip(cy, 0, H)
            for i in range(N):
                r_bound = min(cx[i], W - cx[i], cy[i], H - cy[i])
                r[i] = min(r[i], r_bound)
    
    # Final validation
    out = np.column_stack([best_cx, best_cy, best_r])
    valid, final_score = _validate(out)
    if not valid:
        return None, 0.0
    return out, final_score

def construct():
    start_time = time.time()
    
    best_score = 0.0
    best_out = None
    
    # Aspect ratios to try: W + H = 2
    # Square is 1x1. Rectangle can be 0.6x1.4 to 1.4x0.6
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.1, 0.9]
    
    layouts_per_aspect = 3  # Grid, Hex, Random
    
    for W in aspect_ratios:
        if not _time_ok():
            break
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        time_budget = min(10.0, DEADLINE - time.time() - 5.0)
        if time_budget < 1.0:
            break
        
        for layout_idx in range(layouts_per_aspect):
            if not _time_ok():
                break
            
            if layout_idx == 0:
                # Grid
                kx, ky = 4, 5  # 20 circles, add 1 more
                x0 = _make_grid(kx, ky, W, H)
            elif layout_idx == 1:
                # Hex
                x0 = _hexagonal_layout(W, H)
            else:
                # Random
                x0 = _random_layout(W, H)
            
            result, score = _optimize_one(W, H, x0, time_budget)
            
            if result is not None and score > best_score:
                best_score = score
                best_out = result.copy()
                
                # If we get a very good score, we can stop early
                if best_score > 2.2: # Example threshold, sum of 21 radii
                    break
        
        if best_score > 2.2:
            break

    # If no valid result, fall back to simple grid
    if best_out is None:
        W, H = 1.0, 1.0
        x0 = _make_grid(5, 5, W, H)
        cx, cy, r = x0[:, 0], x0[:, 1], x0[:, 2]
        cx, cy, r = _relax(cx, cy, r, W, H, n_iter=20)
        out = np.column_stack([cx, cy, r])
        valid, score = _validate(out)
        if valid:
            return out
        return x0

    return best_out

# EVOLVE-BLOCK-START
# (The entire implementation is above; this block marker is for compatibility)
# EVOLVE-BLOCK-END
