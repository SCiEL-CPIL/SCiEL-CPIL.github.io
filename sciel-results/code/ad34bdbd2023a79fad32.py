"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _force_directed_packing(N, w, h, n_iters=8000, lr_start=0.005, lr_end=0.00005,
                             n_restarts=3, seed=None):
    """
    Force-directed layout for circle packing in a w x h rectangle.
    Uses contact graph springs and wall repulsion.
    Returns (x, y, r) arrays.
    """
    if seed is not None:
        np.random.seed(seed)
    
    best_sum_r = -np.inf
    best_x, best_y, best_r = None, None, None
    
    for restart in range(n_restarts):
        # Initialize with perturbed grid layout
        cols = int(np.ceil(np.sqrt(N * w / h)))
        rows = int(np.ceil(N / cols))
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            col = i % cols
            row = i // cols
            base_x = (col + 0.5) * w / cols
            base_y = (row + 0.5) * h / rows
            # Add small random perturbation for diversity
            x[i] = base_x + np.random.randn() * 0.02 * w
            y[i] = base_y + np.random.randn() * 0.02 * h
            r[i] = min(w / cols, h / rows) * 0.35
        
        # Force-directed relaxation with contact graph
        for it in range(n_iters):
            t = it / n_iters
            # Exponential decay of learning rate
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
                    
                    if dist < 1e-12:
                        dx = np.random.randn() * 1e-7
                        dy = np.random.randn() * 1e-7
                        dist = np.sqrt(dx * dx + dy * dy)
                    
                    if dist < min_dist:
                        # Overlapping: strong repulsive force
                        overlap = min_dist - dist
                        force = 15.0 * overlap
                        fx[i] += force * dx / dist
                        fy[i] += force * dy / dist
                        fx[j] -= force * dx / dist
                        fy[j] -= force * dy / dist
                    elif dist < min_dist * 1.02:
                        # Near contact: spring force to maintain exact distance
                        diff = dist - min_dist
                        force = 2.0 * diff
                        fx[i] += force * dx / dist
                        fy[i] += force * dy / dist
                        fx[j] -= force * dx / dist
                        fy[j] -= force * dy / dist
            
            # Boundary forces (stronger for better wall contact)
            for i in range(N):
                if x[i] < r[i]:
                    penetration = r[i] - x[i]
                    fx[i] += 8.0 * penetration
                if x[i] > w - r[i]:
                    penetration = x[i] - (w - r[i])
                    fx[i] -= 8.0 * penetration
                if y[i] < r[i]:
                    penetration = r[i] - y[i]
                    fy[i] += 8.0 * penetration
                if y[i] > h - r[i]:
                    penetration = y[i] - (h - r[i])
                    fy[i] -= 8.0 * penetration
            
            # Update positions
            x += lr * fx
            y += lr * fy
            
            # Clamp to valid range
            for i in range(N):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])
        
        # Grow radii iteratively
        for grow_iter in range(100):
            r_new = r * 1.005
            valid = True
            
            for i in range(N):
                if x[i] < r_new[i] or x[i] > w - r_new[i]:
                    valid = False
                    break
                if y[i] < r_new[i] or y[i] > h - r_new[i]:
                    valid = False
                    break
            
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist < r_new[i] + r_new[j] - 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                r = r_new
            else:
                break
        
        # Validate and track best
        if _validate_packing(x, y, r, w, h):
            sum_r = np.sum(r)
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    if best_x is not None:
        return best_x, best_y, best_r
    
    # Fallback: return initial grid
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    for i in range(N):
        col = i % cols
        row = i // cols
        x[i] = (col + 0.5) * w / cols
        y[i] = (row + 0.5) * h / rows
        r[i] = min(w / cols, h / rows) * 0.3
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
    Use force-directed contact graph layout with multiple aspect ratios.
    """
    best_val = -1
    best_result = None
    
    # Try multiple aspect ratios with finer granularity
    # Perimeter = 2(w+h) = 4, so w+h = 2
    aspect_ratios = np.linspace(0.7, 1.3, 10)
    
    for w in aspect_ratios:
        h = 2.0 - w
        
        # Run force-directed packing with multiple restarts
        x, y, r = _force_directed_packing(N, w, h, n_iters=5000, n_restarts=3)
        
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
