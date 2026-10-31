"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _time_ok(deadline, margin=0.5):
    return time.time() < deadline - margin

def _compute_rect(width):
    """Given width, compute height from perimeter=4 constraint: 2*(w+h)=4 => h=2-w"""
    height = 2.0 - width
    return width, height

def _objective(circles, width):
    """
    Compute the sum of radii, with penalty if circles overlap or go out of bounds.
    circles: array of shape (N, 3) with [x, y, r]
    Returns: (value, is_valid)
    """
    height = 2.0 - width
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check bounds
    if np.any(x < 0) or np.any(x > width) or np.any(y < 0) or np.any(y > height):
        return 0.0, False
    
    # Check for overlaps
    n = len(circles)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j] - 1e-10:
                return 0.0, False
    
    return np.sum(r), True

def _force_directed(circles, width, max_iters=200, learning_rate=0.001):
    """
    Force-directed relaxation to find better packing.
    circles: array of shape (N, 3) with [x, y, r]
    """
    height = 2.0 - width
    circles = circles.copy()
    n = len(circles)
    
    for iteration in range(max_iters):
        forces = np.zeros_like(circles)
        
        # Boundary forces
        for i in range(n):
            x, y, r = circles[i]
            # Push away from boundaries
            if x < r:
                forces[i, 0] += (r - x) * 10
            elif x > width - r:
                forces[i, 0] -= (x - (width - r)) * 10
            if y < r:
                forces[i, 1] += (r - y) * 10
            elif y > height - r:
                forces[i, 1] -= (y - (height - r)) * 10
        
        # Pairwise repulsion for overlapping circles
        for i in range(n):
            for j in range(i+1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = circles[i, 2] + circles[j, 2]
                
                if dist < min_dist and dist > 1e-10:
                    # Repulsive force
                    overlap = min_dist - dist
                    fx = (dx / dist) * overlap * 5
                    fy = (dy / dist) * overlap * 5
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Update positions
        circles[:, 0] += learning_rate * forces[:, 0]
        circles[:, 1] += learning_rate * forces[:, 1]
        
        # Clamp to bounds
        circles[:, 0] = np.clip(circles[:, 0], 0, width)
        circles[:, 1] = np.clip(circles[:, 1], 0, height)
        
        # Decay learning rate
        learning_rate *= 0.995
    
    return circles

def _local_refine(circles, width, max_iters=100):
    """
    Local refinement using gradient descent on radius.
    """
    height = 2.0 - width
    circles = circles.copy()
    n = len(circles)
    x = circles[:, 0]
    y = circles[:, 1]
    
    for iteration in range(max_iters):
        # Compute max possible radius for each circle
        new_r = np.zeros(n)
        for i in range(n):
            # Distance to boundaries
            dist_bounds = min(x[i], y[i], width - x[i], height - y[i])
            
            # Distance to other circles
            min_dist = dist_bounds
            for j in range(n):
                if i != j:
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    min_dist = min(min_dist, d - circles[j, 2])
            
            new_r[i] = max(0.001, min_dist)
        
        # Update radii with damping
        circles[:, 2] = 0.7 * circles[:, 2] + 0.3 * new_r
    
    return circles

def _grid_layout(width):
    """Create a grid layout for initialization."""
    height = 2.0 - width
    aspect = width / height
    side = int(np.ceil(np.sqrt(N * aspect)))
    
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) * width / side
            y = (j + 0.5) * height / side
            r = min(width / side, height / side) / 2 * 0.9
            cells.append([x, y, r])
    
    return np.array(cells[:N])

def _random_layout(width):
    """Create a random layout for initialization."""
    height = 2.0 - width
    circles = np.zeros((N, 3))
    circles[:, 0] = np.random.uniform(0.05, width - 0.05, N)
    circles[:, 1] = np.random.uniform(0.05, height - 0.05, N)
    circles[:, 2] = np.random.uniform(0.02, 0.1, N)
    return circles

def construct():
    deadline = time.time() + 145.0
    best_circles = None
    best_value = 0.0
    
    # Try multiple width values
    widths = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6]
    
    for width in widths:
        if not _time_ok(deadline, 1.0):
            break
        
        height = 2.0 - width
        if height <= 0:
            continue
        
        # Try grid initialization
        if _time_ok(deadline, 1.0):
            circles = _grid_layout(width)
            circles = _force_directed(circles, width, max_iters=150)
            circles = _local_refine(circles, width, max_iters=80)
            value, valid = _objective(circles, width)
            if valid and value > best_value:
                best_value = value
                best_circles = circles.copy()
        
        # Try random initialization (a few attempts)
        for attempt in range(3):
            if not _time_ok(deadline, 1.0):
                break
            
            circles = _random_layout(width)
            circles = _force_directed(circles, width, max_iters=100)
            circles = _local_refine(circles, width, max_iters=60)
            value, valid = _objective(circles, width)
            if valid and value > best_value:
                best_value = value
                best_circles = circles.copy()
    
    # Final refinement on best solution
    if best_circles is not None and _time_ok(deadline, 2.0):
        # Re-optimize width around best
        best_width = None
        for w in np.linspace(0.5, 1.5, 21):
            if not _time_ok(deadline, 2.0):
                break
            h = 2.0 - w
            if h <= 0:
                continue
            # Check if current circles fit in this rectangle
            if np.all(best_circles[:, 0] <= w) and np.all(best_circles[:, 1] <= h):
                if best_width is None or w < best_width:
                    best_width = w
        
        if best_width is not None:
            # Final force-directed pass
            best_circles = _force_directed(best_circles, best_width, max_iters=100)
            best_circles = _local_refine(best_circles, best_width, max_iters=100)
    
    if best_circles is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
