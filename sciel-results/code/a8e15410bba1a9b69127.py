"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _relax_layout(x, y, r, w, h, n_iters=2000, lr_start=0.01, lr_end=0.0001):
    """
    Relax the layout to resolve overlaps and push circles to boundaries.
    Returns updated x, y.
    """
    for it in range(n_iters):
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
                
                if dist_sq < 1e-10:
                    dx = np.random.randn() * 1e-6
                    dy = np.random.randn() * 1e-6
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq)
                else:
                    dist = np.sqrt(dist_sq)
                
                if dist < min_dist:
                    # Overlap: strong repulsion
                    overlap = min_dist - dist
                    force = 10.0 * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                # We do not add attractive forces for near-contacts here to avoid
                # sticking, but the boundary forces will push them out.
        
        # Boundary forces
        for i in range(N):
            if x[i] < r[i]:
                penetration = r[i] - x[i]
                fx[i] += 5.0 * penetration
            if x[i] > w - r[i]:
                penetration = x[i] - (w - r[i])
                fx[i] -= 5.0 * penetration
            if y[i] < r[i]:
                penetration = r[i] - y[i]
                fy[i] += 5.0 * penetration
            if y[i] > h - r[i]:
                penetration = y[i] - (h - r[i])
                fy[i] -= 5.0 * penetration
        
        # Update positions
        x += lr * fx
        y += lr * fy
        
        # Clamp to valid range
        for i in range(N):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
            
    return x, y

def _grow_and_relax(w, h, n_trials=3):
    """
    Attempt to pack N circles in w x h by growing radii.
    Returns best sum of radii and the corresponding configuration.
    """
    best_val = -1
    best_config = None
    
    for trial in range(n_trials):
        # Initialize positions
        cols = int(np.ceil(np.sqrt(N)))
        rows = int(np.ceil(N / cols))
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            col = i % cols
            row = i // cols
            # Add jitter to break symmetry
            x[i] = (col + 0.5) * w / cols + np.random.uniform(-0.05, 0.05) * (w/cols)
            y[i] = (row + 0.5) * h / rows + np.random.uniform(-0.05, 0.05) * (h/rows)
            # Start with small radii
            r[i] = 0.01
            
        # Grow and Relax Loop
        growth_factor = 1.01
        max_growth_steps = 200
        
        for step in range(max_growth_steps):
            # Try to grow
            r_new = r * growth_factor
            
            # Check if circles can fit in bounds with new radii
            # If a circle is too big for the box, we stop
            if np.any(r_new > w/2) or np.any(r_new > h/2):
                break
                
            # Relax with new radii
            x_relaxed, y_relaxed = _relax_layout(x, y, r_new, w, h, n_iters=500, lr_start=0.005)
            
            # Check validity
            valid = True
            for i in range(N):
                if x_relaxed[i] - r_new[i] < -1e-6 or x_relaxed[i] + r_new[i] > w + 1e-6:
                    valid = False
                    break
                if y_relaxed[i] - r_new[i] < -1e-6 or y_relaxed[i] + r_new[i] > h + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = x_relaxed[i] - x_relaxed[j]
                        dy = y_relaxed[i] - y_relaxed[j]
                        dist_sq = dx * dx + dy * dy
                        min_dist = r_new[i] + r_new[j]
                        if dist_sq < min_dist * min_dist - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                x = x_relaxed
                y = y_relaxed
                r = r_new
                # Optionally reduce growth factor slightly as we get larger
                # growth_factor *= 0.995 
            else:
                # Failed to fit, try smaller growth or break
                # If we just started growing, maybe relax more? 
                # For simplicity, we break and accept current state
                break
        
        val = np.sum(r)
        if val > best_val:
            best_val = val
            best_config = np.column_stack([x, y, r])
            
    return best_val, best_config

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
    Use grow-and-relax force-directed layout.
    """
    best_val = -1
    best_result = None
    
    # Perimeter = 2(w+h) = 4 => w+h = 2
    # Try a range of aspect ratios
    # Square is often good, but slightly rectangular can be better for odd N
    aspect_ratios = np.linspace(0.7, 1.3, 13)
    
    for w in aspect_ratios:
        h = 2.0 - w
        
        val, config = _grow_and_relax(w, h, n_trials=2)
        
        if config is not None:
            x, y, r = config[:, 0], config[:, 1], config[:, 2]
            if _validate_packing(x, y, r, w, h):
                if val > best_val:
                    best_val = val
                    best_result = config
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid in 1x1 square
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
