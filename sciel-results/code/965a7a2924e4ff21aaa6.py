"""Combined program: Force-directed relaxation for circle packing."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    # Initialize with a valid grid packing in a 1x1 square
    # 21 circles in a 5x5 grid (using 21 cells)
    # Cell size 0.2, radius 0.09 (small enough to fit)
    best_result = None
    best_score = -np.inf

    def create_grid_init(w):
        h = 2.0 - w
        # Determine grid dimensions
        # Try to balance aspect ratio
        aspect = w / h
        if aspect >= 1.0:
            n_cols = int(np.round(np.sqrt(N * aspect)))
            n_rows = int(np.ceil(N / n_cols))
        else:
            n_rows = int(np.round(np.sqrt(N / aspect)))
            n_cols = int(np.ceil(N / n_rows))
        
        # Adjust to ensure N cells
        while n_cols * n_rows < N:
            if aspect >= 1.0:
                n_cols += 1
            else:
                n_rows += 1
            
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        cell_w = w / n_cols
        cell_h = h / n_rows
        max_r = min(cell_w, cell_h) * 0.45  # Leave some gap
        
        idx = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if idx >= N:
                    break
                x[idx] = (j + 0.5) * cell_w
                y[idx] = (i + 0.5) * cell_h
                r[idx] = max_r
                idx += 1
        return x, y, r

    def validate(w, h, x, y, r, tol=1e-9):
        if w < 0.1 or h < 0.1:
            return False
        if np.any(r < 0):
            return False
        if np.any(x - r < -tol) or np.any(w - x - r < -tol):
            return False
        if np.any(y - r < -tol) or np.any(h - y - r < -tol):
            return False
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                if dist_sq < (r[i] + r[j] - tol) ** 2:
                    return False
        return True

    def update_best(w, h, x, y, r):
        nonlocal best_result, best_score
        if validate(w, h, x, y, r):
            # Clip small numerical errors
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)
            r = np.clip(r, 0, None)
            
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x, y, r])
                return True
        return False

    # Initial guess: Grid in square
    w_init = 1.0
    h_init = 1.0
    x, y, r = create_grid_init(w_init)
    update_best(w_init, h_init, x, y, r)

    # Force-Directed Relaxation
    # We optimize w, x, y, r.
    # Forces:
    # 1. Repulsion between overlapping circles.
    # 2. Repulsion from walls if circle goes out.
    # 3. Attraction to increase radius (we want to maximize sum r).
    #    Instead of direct attraction, we can just let the repulsion push them apart,
    #    and then we can increase radii slightly if there is space?
    #    Actually, a common technique is to fix radii and optimize positions,
    #    or fix positions and optimize radii.
    #    Let's do a simple iterative solver:
    #    1. Start with small radii.
    #    2. Increase radii slightly.
    #    3. Relax positions to remove overlaps (force-directed).
    #    4. Repeat.
    
    # Parameters
    max_iter = 500
    step_size = 0.05
    radius_growth = 0.01
    
    # Current state
    w = w_init
    h = h_init
    x, y, r = create_grid_init(w)
    # Start with smaller radii to allow movement
    r = r * 0.5
    
    # Pre-allocate force arrays
    fx = np.zeros(N)
    fy = np.zeros(N)
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
            
        # 1. Calculate forces
        fx.fill(0.0)
        fy.fill(0.0)
        
        # Circle-Circle Repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq)
                min_dist = r[i] + r[j]
                
                if dist < min_dist and dist > 1e-6:
                    # Force proportional to overlap
                    overlap = min_dist - dist
                    force = overlap * 10.0  # Stiffness
                    # Normalize direction
                    fx[i] += force * (dx / dist)
                    fy[i] += force * (dy / dist)
                    fx[j] -= force * (dx / dist)
                    fy[j] -= force * (dy / dist)
                elif dist <= 1e-6:
                    # Jitter if exactly same position
                    fx[i] += np.random.uniform(-1, 1)
                    fy[i] += np.random.uniform(-1, 1)
                    fx[j] -= np.random.uniform(-1, 1)
                    fy[j] -= np.random.uniform(-1, 1)

        # Wall Repulsion
        for i in range(N):
            # Left
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 10.0
            # Right
            if w - x[i] < r[i]:
                fx[i] -= (r[i] - (w - x[i])) * 10.0
            # Bottom
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 10.0
            # Top
            if h - y[i] < r[i]:
                fy[i] -= (r[i] - (h - y[i])) * 10.0

        # 2. Update positions
        # Damping to prevent oscillation
        x += fx * step_size * 0.1
        y += fy * step_size * 0.1
        
        # 3. Clamp positions to valid range (soft clamp handled by forces, hard clamp for stability)
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)
        
        # 4. Try to increase radii
        # Check if current configuration is valid with slightly larger radii
        new_r = r * (1.0 + radius_growth)
        
        # Validate with new radii
        if validate(w, h, x, y, new_r, tol=1e-8):
            r = new_r
        else:
            # If not valid, maybe reduce growth or keep same
            # We can also try to optimize w
            pass

        # 5. Occasionally try to optimize w
        # Simple heuristic: if aspect ratio is far from 1, adjust?
        # Or just try a few fixed w values in a loop?
        # Let's try to adjust w to better fit the aspect ratio of the point cloud
        # Calculate bounding box of centers
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        span_x = x_max - x_min + 2 * np.mean(r)
        span_y = y_max - y_min + 2 * np.mean(r)
        
        if span_x > 0 and span_y > 0:
            target_aspect = span_x / span_y
            # Target w such that w/h = target_aspect and w+h=2
            # w/(2-w) = A => w = 2A / (1+A)
            target_w = 2.0 * target_aspect / (1.0 + target_aspect)
            # Clamp target_w
            target_w = np.clip(target_w, 0.5, 1.5)
            
            # Move w towards target
            w = 0.9 * w + 0.1 * target_w
            h = 2.0 - w
            
            # Re-clamp positions to new bounds
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)
            
            # If w changed, radii might now be invalid, check
            if not validate(w, h, x, y, r, tol=1e-8):
                 # Reduce radii to fit
                 r = r * 0.95

        # Update best
        update_best(w, h, x, y, r)

    # Final Polish: Try to squeeze in a bit more by reducing tolerance in validation
    # and doing a few more iterations with smaller steps
    step_size = 0.01
    for _ in range(100):
        if time.time() > deadline - 1.0:
            break
        fx.fill(0.0)
        fy.fill(0.0)
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-6:
                    overlap = min_dist - dist
                    force = overlap * 10.0
                    fx[i] += force * (dx / dist)
                    fy[i] += force * (dy / dist)
                    fx[j] -= force * (dx / dist)
                    fy[j] -= force * (dy / dist)
        for i in range(N):
            if x[i] < r[i]: fx[i] += (r[i] - x[i]) * 10.0
            if w - x[i] < r[i]: fx[i] -= (r[i] - (w - x[i])) * 10.0
            if y[i] < r[i]: fy[i] += (r[i] - y[i]) * 10.0
            if h - y[i] < r[i]: fy[i] -= (r[i] - (h - y[i])) * 10.0
            
        x += fx * step_size * 0.1
        y += fy * step_size * 0.1
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)
        
        # Try increasing radii
        new_r = r * 1.001
        if validate(w, h, x, y, new_r, tol=1e-9):
            r = new_r
            
        update_best(w, h, x, y, r)

    # Fallback if best_result is still None (should not happen)
    if best_result is None:
        x, y, r = create_grid_init(1.0)
        best_result = np.column_stack([x, y, r])

    return best_result
# EVOLVE-BLOCK-END
