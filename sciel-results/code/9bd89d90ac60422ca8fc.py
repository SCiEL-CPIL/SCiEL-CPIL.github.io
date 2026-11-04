"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    circles = np.asarray(circles)
    n = len(circles)
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    for i in range(n):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(dist < r[i] + r[i+1:] - tol):
            return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _make_hexagonal_grid(n, w, h):
    """Create initial hexagonal-like arrangement for 21 circles."""
    rows = [5, 4, 5, 4, 3]
    circles = []
    total_rows = len(rows)
    
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / total_rows
        for col_j in range(count):
            # Offset alternating rows for hexagonal packing
            offset = 0.5 * (w / 5) if row_i % 2 == 1 else 0
            x = (col_j + 0.5) * w / 5 + offset
            r = 0.35 * min(w/5, h/total_rows)
            circles.append([x, y, r])
    
    return np.array(circles[:n])

def _make_grid(n, w, h):
    """Fallback grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            r = 0.5 * min(w, h) / side * 0.9
            cells.append([cx, cy, r])
    return np.array(cells[:n])

def _relax_circles(circles, w, h, iterations=1000, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    Vectorized for speed.
    """
    n = len(circles)
    pos = np.array(circles[:, :2], dtype=float)
    rad = np.array(circles[:, 2], dtype=float)

    for _ in range(iterations):
        moved = False

        # 1. Resolve boundary violations
        for i in range(n):
            if pos[i, 0] - rad[i] < 0:
                pos[i, 0] = rad[i]
                moved = True
            if pos[i, 0] + rad[i] > w:
                pos[i, 0] = w - rad[i]
                moved = True
            if pos[i, 1] - rad[i] < 0:
                pos[i, 1] = rad[i]
                moved = True
            if pos[i, 1] + rad[i] > h:
                pos[i, 1] = h - rad[i]
                moved = True

        # 2. Resolve overlaps (vectorized per circle)
        for i in range(n):
            dx = pos[i+1:, 0] - pos[i, 0]
            dy = pos[i+1:, 1] - pos[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rad[i] + rad[i+1:]
            overlap = min_dist - dist
            
            for k in range(len(overlap)):
                if overlap[k] > 0:
                    if dist[k] < 1e-9:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = overlap[k] / 2.0
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[i+1+k, 0] += np.cos(angle) * push
                        pos[i+1+k, 1] += np.sin(angle) * push
                    else:
                        push = overlap[k] / 2.0
                        ux = dx[k] / dist[k]
                        uy = dy[k] / dist[k]
                        pos[i, 0] -= ux * push
                        pos[i, 1] -= uy * push
                        pos[i+1+k, 0] += ux * push
                        pos[i+1+k, 1] += uy * push
                    moved = True

        if not moved:
            break

    # Final boundary clamp
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])

    new_circles = np.column_stack((pos, rad))

    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles

    return circles

def _radial_chain_optimize(circles, w, h, deadline):
    """
    Structural optimization: treat the packing as a rigid body with a global scale factor.
    We optimize a single scalar 'scale' that multiplies all radii and scales the positions
    relative to the center, while maintaining the relative geometry from the relaxation step.
    This reduces the problem from 63 DOF to 1 DOF, allowing for a very precise fit.
    """
    if time.time() > deadline - 2.0:
        return None

    n = len(circles)
    # Center the configuration
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    
    # Compute center of mass (weighted by radius? or just geometric center?)
    # Let's use the bounding box center for scaling stability
    min_x = np.min(cx - r)
    max_x = np.max(cx + r)
    min_y = np.min(cy - r)
    max_y = np.max(cy + r)
    
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    
    # Relative positions
    rx = cx - center_x
    ry = cy - center_y
    
    def objective(scale):
        new_r = r * scale
        new_cx = center_x + rx * scale
        new_cy = center_y + ry * scale
        
        # Check bounds
        if np.any(new_cx - new_r < -1e-9) or np.any(new_cx + new_r > w + 1e-9):
            return 0.0
        if np.any(new_cy - new_r < -1e-9) or np.any(new_cy + new_r > h + 1e-9):
            return 0.0
        
        # Check overlaps
        for i in range(n):
            dx = new_cx[i] - new_cx[i+1:]
            dy = new_cy[i] - new_cy[i+1:]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = new_r[i] + new_r[i+1:]
            if np.any(dist < min_dist - 1e-9):
                return 0.0
        
        return np.sum(new_r)
    
    # Bisection search for the maximum valid scale
    # The current configuration is valid (scale=1). We can try to increase it.
    # But we might also be able to decrease it if it's invalid? 
    # The relaxation step should have made it valid.
    # So we start with scale=1 and try to increase.
    
    lo = 1.0
    hi = 1.5  # Try up to 50% larger
    
    # First check if we can even increase
    if objective(hi) == 0.0:
        # Binary search for the largest valid scale between 1.0 and hi
        for _ in range(30):
            mid = (lo + hi) / 2.0
            if objective(mid) > 0:
                lo = mid
            else:
                hi = mid
        best_scale = lo
    else:
        # It's still valid at 1.5, maybe we can go higher?
        # Let's just stick with 1.5 or try a bit higher, but 1.5 is a good cap for safety.
        best_scale = hi

    if best_scale <= 1.0:
        return circles

    new_circles = np.column_stack((
        center_x + rx * best_scale,
        center_y + ry * best_scale,
        r * best_scale
    ))
    
    if _validate(new_circles, w, h):
        return new_circles
    
    return circles

def _local_optimize(circles, w, h, deadline):
    """
    Local optimization using Nelder-Mead to maximize sum of radii.
    Variables: (cx_i, cy_i, r_i) for each circle i.
    """
    n = len(circles)
    x0 = []
    for cx, cy, r in circles:
        x0.extend([cx, cy, r])
    x0 = np.array(x0)
    
    def objective(x):
        if time.time() > deadline:
            return 0.0
        circles = []
        for i in range(n):
            cx = x[i * 3]
            cy = x[i * 3 + 1]
            r = x[i * 3 + 2]
            circles.append((cx, cy, r))
        
        # Penalty-based objective: return negative sum of radii if invalid
        for cx, cy, r in circles:
            if r <= 0:
                return 0.0
            if cx - r < -1e-9 or cx + r > w + 1e-9 or cy - r < -1e-9 or cy + r > h + 1e-9:
                return 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < circles[i][2] + circles[j][2] - 1e-9:
                    return 0.0
        
        return _total_radius(circles)
    
    try:
        res = minimize(
            lambda x: -objective(x),
            x0,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if objective(res.x) > 0:
            best_circles = []
            for i in range(n):
                best_circles.append((res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]))
            if _validate(best_circles, w, h):
                return np.array(best_circles)
    except Exception:
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Perimeter 4 => 2(w+h) = 4 => w+h = 2
    # Try various aspect ratios
    best_circles = None
    best_score = 0.0
    
    # Aspect ratios to try: (w, h) where w+h=2
    # w ranges from 0.5 to 1.5 (symmetric, so w<=h means w<=1)
    aspect_ratios = [
        (1.0, 1.0),    # square
        (0.8, 1.2),
        (0.7, 1.3),
        (0.6, 1.4),
        (0.9, 1.1),
        (0.75, 1.25),
        (0.85, 1.15),
        (0.95, 1.05),
    ]
    
    for w, h in aspect_ratios:
        if time.time() > deadline:
            break
        
        # Create initial hexagonal grid
        circles = _make_hexagonal_grid(N, w, h)
        
        # Validate initial grid
        if not _validate(circles, w, h):
            # Try to fix with relaxation
            circles = _relax_circles(circles, w, h)
            if not _validate(circles, w, h):
                continue
        
        score = _total_radius(circles)
        if score > best_score:
            best_score = score
            best_circles = circles
        
        # Step 1: Relaxation to ensure validity and resolve overlaps
        if time.time() < deadline - 10.0:
            relaxed = _relax_circles(circles, w, h)
            if relaxed is not None and _validate(relaxed, w, h):
                rel_score = _total_radius(relaxed)
                if rel_score > best_score:
                    best_score = rel_score
                    best_circles = relaxed
                circles = relaxed
            
        # Step 2: Radial Chain Scaling (Structural Optimization)
        if time.time() < deadline - 5.0:
            scaled = _radial_chain_optimize(circles, w, h, deadline)
            if scaled is not None:
                sc_score = _total_radius(scaled)
                if sc_score > best_score:
                    best_score = sc_score
                    best_circles = scaled
                circles = scaled

        # Step 3: Local Optimization (Nelder-Mead) for fine-tuning
        if time.time() < deadline - 5.0:
            optimized = _local_optimize(circles, w, h, deadline)
            if optimized is not None:
                opt_score = _total_radius(optimized)
                if opt_score > best_score:
                    best_score = opt_score
                    best_circles = optimized
    
    # If we found a good result, return it
    if best_circles is not None:
        return np.array(best_circles)
    
    # Fallback: simple grid in 1x1 square
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
