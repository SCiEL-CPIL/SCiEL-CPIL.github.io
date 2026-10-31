"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _validate(circles, tol=1e-6):
    """Check validity: disjoint interiors, within rectangle of perimeter <= 4."""
    n = len(circles)
    if n == 0:
        return False
    # Compute bounding box
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    width = max_x - min_x
    height = max_y - min_y
    perimeter = 2 * (width + height)
    if perimeter > 4 + tol:
        return False
    # Check disjoint interiors
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    # Check radii positive
    if np.any(rs <= 0):
        return False
    return True

def _pressure_relaxation(init_circles, max_iter=5000, dt=0.001, deadline=None):
    """
    Particle-based pressure relaxation: circles repel each other and are 
    confined by a virtual rectangle. The rectangle dimensions are adjusted
    to maintain perimeter = 4.
    
    We treat this as a system where:
    - Circles repel each other with force proportional to overlap
    - Circles are pushed back by walls
    - The rectangle shape (aspect ratio) is optimized to maximize sum of radii
    """
    circles = init_circles.copy()
    n = len(circles)
    
    # Start with a square-ish rectangle
    # Perimeter = 2*(w+h) = 4, so w+h = 2
    # Try different aspect ratios
    best_circles = None
    best_value = -1
    
    for aspect in [1.0, 0.8, 1.25, 0.6, 1.667, 0.5]:
        # w * aspect = h, w + h = 2 => w(1 + aspect) = 2 => w = 2/(1+aspect)
        w = 2.0 / (1.0 + aspect)
        h = 2.0 - w
        
        # Reset to initial positions scaled to this rectangle
        c = init_circles.copy()
        # Scale positions to fit in [0,w] x [0,h]
        # First normalize to [0,1]x[0,1]
        min_x = np.min(c[:, 0] - c[:, 2])
        max_x = np.max(c[:, 0] + c[:, 2])
        min_y = np.min(c[:, 1] - c[:, 2])
        max_y = np.max(c[:, 1] + c[:, 2])
        span_x = max_x - min_x if max_x > min_x else 1.0
        span_y = max_y - min_y if max_y > min_y else 1.0
        
        # Scale to fit with some margin
        margin = 0.01
        scale_x = (w - 2 * margin) / span_x
        scale_y = (h - 2 * margin) / span_y
        scale = min(scale_x, scale_y)
        
        c[:, 0] = (c[:, 0] - min_x) * scale + margin + (w - 2 * margin - span_x * scale) / 2
        c[:, 1] = (c[:, 1] - min_y) * scale + margin + (h - 2 * margin - span_y * scale) / 2
        c[:, 2] = c[:, 2] * scale
        
        # Ensure circles fit within rectangle
        for i in range(n):
            c[i, 0] = np.clip(c[i, 0], c[i, 2], w - c[i, 2])
            c[i, 1] = np.clip(c[i, 1], c[i, 2], h - c[i, 2])
        
        # Pressure relaxation iterations
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            forces = np.zeros_like(c)
            
            # Circle-circle repulsion
            for i in range(n):
                for j in range(i + 1, n):
                    dx = c[i, 0] - c[j, 0]
                    dy = c[i, 1] - c[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = c[i, 2] + c[j, 2]
                    if dist < min_dist:
                        if dist > 1e-10:
                            fx = dx / dist
                            fy = dy / dist
                        else:
                            fx = np.random.randn()
                            fy = np.random.randn()
                            norm = np.sqrt(fx * fx + fy * fy)
                            fx /= norm
                            fy /= norm
                        overlap = min_dist - dist
                        force_mag = overlap * 50.0  # Stiff spring
                        forces[i, 0] += fx * force_mag
                        forces[i, 1] += fy * force_mag
                        forces[j, 0] -= fx * force_mag
                        forces[j, 1] -= fy * force_mag
            
            # Wall repulsion (soft)
            wall_strength = 100.0
            for i in range(n):
                # Left wall
                if c[i, 0] < c[i, 2]:
                    forces[i, 0] += wall_strength * (c[i, 2] - c[i, 0])
                # Right wall
                if c[i, 0] > w - c[i, 2]:
                    forces[i, 0] -= wall_strength * (c[i, 0] - (w - c[i, 2]))
                # Bottom wall
                if c[i, 1] < c[i, 2]:
                    forces[i, 1] += wall_strength * (c[i, 2] - c[i, 1])
                # Top wall
                if c[i, 1] > h - c[i, 2]:
                    forces[i, 1] -= wall_strength * (c[i, 1] - (h - c[i, 2]))
            
            # Update positions
            c[:, 0] += dt * forces[:, 0]
            c[:, 1] += dt * forces[:, 1]
            
            # Hard constraint: keep circles inside rectangle
            for i in range(n):
                c[i, 0] = np.clip(c[i, 0], c[i, 2], w - c[i, 2])
                c[i, 1] = np.clip(c[i, 1], c[i, 2], h - c[i, 2])
            
            # Try to expand radii slightly if no overlaps
            # Check if all circles are separated
            min_overlap = 0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = c[i, 0] - c[j, 0]
                    dy = c[i, 1] - c[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    overlap = c[i, 2] + c[j, 2] - dist
                    if overlap > min_overlap:
                        min_overlap = overlap
                # Wall overlaps
                wall_overlap = max(0, c[i, 2] - c[i, 0], c[i, 2] - c[i, 1],
                                   c[i, 2] - (w - c[i, 0]), c[i, 2] - (h - c[i, 1]))
                if wall_overlap > min_overlap:
                    min_overlap = wall_overlap
            
            if min_overlap < 0:
                # No overlaps, try to expand
                expansion = 0.0001 * min(-min_overlap, 0.1)
                c[:, 2] += expansion
                # Re-check walls
                for i in range(n):
                    if c[i, 0] - c[i, 2] < 0:
                        c[i, 0] = c[i, 2]
                    if c[i, 0] + c[i, 2] > w:
                        c[i, 0] = w - c[i, 2]
                    if c[i, 1] - c[i, 2] < 0:
                        c[i, 1] = c[i, 2]
                    if c[i, 1] + c[i, 2] > h:
                        c[i, 1] = h - c[i, 2]
            
            # Damping: reduce dt over time
            dt *= 0.9995
        
        # Evaluate this configuration
        if _validate(c):
            value = np.sum(c[:, 2])
            if value > best_value:
                best_value = value
                best_circles = c.copy()
    
    return best_circles if best_circles is not None else init_circles

def _grid_init(n):
    """Create initial grid layout."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side * 0.9])
    return np.array(cells)

def _random_init(n, w=1.0, h=1.0, seed=None):
    """Create random initial layout."""
    if seed is not None:
        np.random.seed(seed)
    rs = np.random.uniform(0.05, 0.15, n)
    xs = np.random.uniform(rs, w - rs, n)
    ys = np.random.uniform(rs, h - rs, n)
    return np.column_stack([xs, ys, rs])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial grid layout
    init = _grid_init(N)
    
    # Normalize to unit square first
    min_x = np.min(init[:, 0] - init[:, 2])
    max_x = np.max(init[:, 0] + init[:, 2])
    min_y = np.min(init[:, 1] - init[:, 2])
    max_y = np.max(init[:, 1] + init[:, 2])
    span_x = max_x - min_x
    span_y = max_y - min_y
    scale = min(1.0 / span_x, 1.0 / span_y)
    init[:, 0] = (init[:, 0] - min_x) * scale
    init[:, 1] = (init[:, 1] - min_y) * scale
    init[:, 2] = init[:, 2] * scale
    
    # Run pressure relaxation
    result = _pressure_relaxation(init, max_iter=3000, dt=0.005, deadline=deadline)
    
    if result is None:
        return init
    
    # Final validation and cleanup
    if not _validate(result):
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return result
# EVOLVE-BLOCK-END
