"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import random


def _force_directed_packing(N, w, h, n_iters=8000, lr_start=0.05, lr_end=0.0005, seed=0):
    """
    Force-directed layout for circle packing in a w x h rectangle.
    Returns (x, y, r) arrays.
    """
    rng = np.random.default_rng(seed)
    
    # Initialize: Random grid with jitter
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    for i in range(N):
        col = i % cols
        row = i // cols
        # Add jitter to break symmetry
        x[i] = (col + 0.5) * w / cols + rng.uniform(-0.1, 0.1) * (w/cols)
        y[i] = (row + 0.5) * h / rows + rng.uniform(-0.1, 0.1) * (h/rows)
        # Initial radius estimate
        r[i] = min(w / cols, h / rows) * 0.45
    
    # Force-directed relaxation
    # Forces:
    # 1. Repulsion between overlapping circles (very strong)
    # 2. Attraction/Spring to maintain contact for near-touching pairs
    # 3. Repulsion from boundary walls
    # 4. Weak attraction to center to prevent drift (optional, usually not needed if walls are strong)
    
    for it in range(n_iters):
        # Learning rate schedule: exponential decay
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
                min_dist = r[i] + r[j]
                
                if dist_sq < 1e-12:
                    # Random small perturbation
                    dx = rng.normal(0, 1e-5)
                    dy = rng.normal(0, 1e-5)
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq)
                else:
                    dist = np.sqrt(dist_sq)
                
                if dist < min_dist:
                    # Overlapping: strong repulsive force
                    # Force proportional to overlap depth
                    overlap = min_dist - dist
                    # Increase force strength as overlap increases to ensure separation
                    force = 100.0 * overlap 
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                elif dist < min_dist * 1.1:
                    # Near contact: spring force to encourage touching
                    # This helps form rigid contact graphs
                    diff = dist - min_dist
                    # We want to pull them together if they are slightly apart, 
                    # but push if overlapping (handled above).
                    # Here dist > min_dist, so diff > 0. We want to reduce diff to 0.
                    # So force should be attractive (negative relative to separation vector i-j)
                    # Vector i->j is (x[j]-x[i], y[j]-y[i]) = (-dx, -dy)
                    # Force on i should be towards j.
                    force = 2.0 * diff
                    fx[i] -= force * dx / dist
                    fy[i] -= force * dy / dist
                    fx[j] += force * dx / dist
                    fy[j] += force * dy / dist
        
        # Boundary forces
        for i in range(N):
            # Left wall
            if x[i] < r[i]:
                penetration = r[i] - x[i]
                fx[i] += 50.0 * penetration
            # Right wall
            if x[i] > w - r[i]:
                penetration = x[i] - (w - r[i])
                fx[i] -= 50.0 * penetration
            # Bottom wall
            if y[i] < r[i]:
                penetration = r[i] - y[i]
                fy[i] += 50.0 * penetration
            # Top wall
            if y[i] > h - r[i]:
                penetration = y[i] - (h - r[i])
                fy[i] -= 50.0 * penetration
        
        # Update positions
        x += lr * fx
        y += lr * fy
        
        # Clamp to valid range to ensure stability
        for i in range(N):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
    
    # Post-processing: Grow radii
    # We try to increase the radii while maintaining non-overlap.
    # This is a greedy approach: try to scale all radii by a factor, check validity, if valid keep, else revert.
    # A better approach is to find the maximum uniform scale factor.
    
    current_scale = 1.0
    best_scale = 1.0
    
    # Binary search for the maximum scale factor
    low, high = 1.0, 2.0
    for _ in range(30):
        mid = (low + high) / 2
        r_test = r * mid
        valid = True
        # Check boundaries
        for i in range(N):
            if x[i] < r_test[i] or x[i] > w - r_test[i]:
                valid = False
                break
            if y[i] < r_test[i] or y[i] > h - r_test[i]:
                valid = False
                break
        if valid:
            # Check pairs
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < r_test[i] + r_test[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            low = mid
            best_scale = mid
        else:
            high = mid
            
    r = r * best_scale
    
    return x, y, r


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
    Use force-directed contact graph layout with multiple restarts.
    """
    best_val = -1
    best_result = None
    
    # Try multiple aspect ratios
    # Perimeter = 2(w+h) = 4, so w+h = 2
    # Optimal aspect ratio for 21 circles is likely close to 1 (square) or slightly rectangular.
    # Let's scan from 0.7 to 1.3 (w from 0.7 to 1.3, h from 1.3 to 0.7)
    aspect_ratios = np.linspace(0.7, 1.3, 11)
    
    for w in aspect_ratios:
        h = 2.0 - w
        
        # Run force-directed packing with multiple restarts
        for restart in range(3):
            x, y, r = _force_directed_packing(N, w, h, n_iters=5000, seed=restart*100 + int(w*100))
            
            # Validate
            if _validate_packing(x, y, r, w, h):
                val = np.sum(r)
                if val > best_val:
                    best_val = val
                    best_result = np.column_stack([x, y, r])
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
