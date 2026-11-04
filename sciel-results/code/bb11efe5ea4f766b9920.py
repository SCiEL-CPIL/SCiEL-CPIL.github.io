"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _force_directed_packing(N, w, h, n_iters=8000, lr_start=0.05, lr_end=0.0001, seed=0):
    """
    Force-directed layout for circle packing in a w x h rectangle.
    Returns (x, y, r) arrays.
    """
    rng = np.random.RandomState(seed)
    
    # 1. Initialization: Jittered Grid
    # A grid is a good starting point to avoid deep local minima where circles clump.
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # Initial radius estimate: assume they fit in a grid
    # We start with small radii and grow them.
    # Initial radius should be small enough to not overlap in the grid.
    # Grid cell size approx w/cols, h/rows.
    r_init = min(w / cols, h / rows) * 0.4
    
    for i in range(N):
        col = i % cols
        row = i // cols
        x[i] = (col + 0.5) * w / cols + rng.uniform(-0.1, 0.1) * (w/cols)
        y[i] = (row + 0.5) * h / rows + rng.uniform(-0.1, 0.1) * (h/rows)
        r[i] = r_init
        
    # Clamp initial positions
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)

    # 2. Force-Directed Relaxation
    # Forces:
    # 1. Overlap: Strong repulsion (k_overlap * overlap)
    # 2. Contact: Spring force to maintain distance (k_contact * (dist - (ri+rj))) 
    #    This encourages a rigid contact graph.
    # 3. Walls: Repulsion if penetrating.
    
    k_overlap = 10.0
    k_contact = 2.0
    k_wall = 5.0
    contact_tolerance = 1.05 # Factor of (ri+rj) to consider "near contact"

    for it in range(n_iters):
        # Learning rate schedule: Exponential decay
        t = it / n_iters
        lr = lr_start * (lr_end / lr_start) ** t
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Circle-circle interactions
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq)
                min_dist = r[i] + r[j]
                
                if dist < 1e-10:
                    # Random small perturbation
                    dx = rng.randn() * 1e-6
                    dy = rng.randn() * 1e-6
                    dist = np.sqrt(dx * dx + dy * dy)
                
                if dist < min_dist:
                    # Overlapping: strong repulsive force
                    overlap = min_dist - dist
                    force = k_overlap * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                elif dist < min_dist * contact_tolerance:
                    # Near contact: spring force
                    # We want to push them towards exactly min_dist.
                    # If dist < min_dist, force is repulsive (handled above).
                    # If dist > min_dist, force is attractive (pulls them together).
                    # This creates a "sticky" effect for potential contacts.
                    diff = dist - min_dist
                    force = k_contact * diff
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Boundary forces
        for i in range(N):
            # Left wall
            if x[i] < r[i]:
                penetration = r[i] - x[i]
                fx[i] += k_wall * penetration
            # Right wall
            if x[i] > w - r[i]:
                penetration = x[i] - (w - r[i])
                fx[i] -= k_wall * penetration
            # Bottom wall
            if y[i] < r[i]:
                penetration = r[i] - y[i]
                fy[i] += k_wall * penetration
            # Top wall
            if y[i] > h - r[i]:
                penetration = y[i] - (h - r[i])
                fy[i] -= k_wall * penetration
        
        # Update positions
        x += lr * fx
        y += lr * fy
        
        # Clamp to valid range to prevent divergence
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # 3. Radius Growth Phase
    # Iteratively increase radii and re-relax.
    # We stop if we can't grow anymore without violation.
    
    current_r = r.copy()
    max_growth_iterations = 100
    growth_step = 0.01 # 1% growth per step
    
    for _ in range(max_growth_iterations):
        # Try to grow
        new_r = current_r * (1.0 + growth_step)
        
        # Check if this growth is feasible with current positions (approximate)
        # A more robust way is to just set radii and run a short relaxation,
        # then check validity. If invalid, revert.
        
        x_test = x.copy()
        y_test = y.copy()
        r_test = new_r.copy()
        
        # Short relaxation with new radii
        # Use a smaller number of iterations for the growth phase
        short_iters = 500
        for it in range(short_iters):
            t = it / short_iters
            lr = 0.02 * (0.001 / 0.02) ** t
            
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x_test[i] - x_test[j]
                    dy = y_test[i] - y_test[j]
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq)
                    min_dist = r_test[i] + r_test[j]
                    
                    if dist < 1e-10:
                        dx = rng.randn() * 1e-6
                        dy = rng.randn() * 1e-6
                        dist = np.sqrt(dx * dx + dy * dy)
                    
                    if dist < min_dist:
                        overlap = min_dist - dist
                        force = 10.0 * overlap
                        fx[i] += force * dx / dist
                        fy[i] += force * dy / dist
                        fx[j] -= force * dx / dist
                        fy[j] -= force * dy / dist
                    elif dist < min_dist * 1.05:
                        diff = dist - min_dist
                        force = 2.0 * diff
                        fx[i] += force * dx / dist
                        fy[i] += force * dy / dist
                        fx[j] -= force * dx / dist
                        fy[j] -= force * dy / dist
            
            for i in range(N):
                if x_test[i] < r_test[i]:
                    fx[i] += 5.0 * (r_test[i] - x_test[i])
                if x_test[i] > w - r_test[i]:
                    fx[i] -= 5.0 * (x_test[i] - (w - r_test[i]))
                if y_test[i] < r_test[i]:
                    fy[i] += 5.0 * (r_test[i] - y_test[i])
                if y_test[i] > h - r_test[i]:
                    fy[i] -= 5.0 * (y_test[i] - (h - r_test[i]))
            
            x_test += lr * fx
            y_test += lr * fy
            x_test = np.clip(x_test, r_test, w - r_test)
            y_test = np.clip(y_test, r_test, h - r_test)
            
        # Validate
        if _validate_packing(x_test, y_test, r_test, w, h, tol=1e-5):
            x = x_test
            y = y_test
            current_r = r_test
        else:
            break
            
    return x, y, current_r


def _validate_packing(x, y, r, w, h, tol=1e-6):
    """Check if packing is valid."""
    for i in range(len(x)):
        if r[i] < -tol:
            return False
        if x[i] - r[i] < -tol or x[i] + r[i] > w + tol:
            return False
        if y[i] - r[i] < -tol or y[i] + r[i] > h + tol:
            return False
    
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - tol:
                return False
    
    return True


def construct():
    """
    Pack 21 circles in a rectangle of perimeter at most 4.
    Use force-directed contact graph layout.
    """
    best_val = -1
    best_result = None
    
    # Time budget
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Try multiple aspect ratios
    # Perimeter = 2(w+h) = 4, so w+h = 2
    # We explore a range of aspect ratios.
    # Symmetry: w and h are interchangeable, so w in [1, 1.5] covers w/h in [1, 2.5] and h/w in [2.5, 1]
    # Actually w+h=2. If w=1, h=1 (square). If w=1.2, h=0.8.
    # Let's sample w from 0.8 to 1.2 (aspect ratio 1.5 to 1/1.5)
    aspect_ratios = np.linspace(0.8, 1.2, 10)
    
    # Multi-start for each aspect ratio
    n_starts = 3
    
    for w in aspect_ratios:
        if time.time() - start_time > time_limit:
            break
        h = 2.0 - w
        
        for seed in range(n_starts):
            if time.time() - start_time > time_limit:
                break
            try:
                x, y, r = _force_directed_packing(N, w, h, n_iters=5000, seed=seed)
            except Exception:
                continue
                
            # Validate
            if _validate_packing(x, y, r, w, h):
                val = np.sum(r)
                if val > best_val:
                    best_val = val
                    best_result = np.column_stack([x, y, r])
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid (should not happen if force-directed works)
    side = int(np.ceil(np.sqrt(N)))
    w = 1.0
    h = 1.0
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            r = 0.5 / side - 1e-3
            cells.append([cx, cy, r])
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
