"""Optimized packing of 50 equal spheres in a unit cube."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 120.0  # seconds, safety margin from 150s limit
TIME_MARGIN = 5.0  # return this many seconds before deadline


def _radius_from_centers(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        return 0.0
    # Boundary constraint: min distance from center to cube face
    dist_to_face = np.min(0.5 - np.abs(centers), axis=1)
    r_boundary = np.min(dist_to_face)
    
    # Pairwise constraint: min distance between any two centers / 2
    # Use scipy's distance matrix for efficiency
    from scipy.spatial.distance import cdist
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    r_pair = np.min(d) / 2.0
    
    return min(r_boundary, r_pair)


def _neg_radius(x):
    """Negative radius to maximize via minimization."""
    centers = x.reshape(-1, 3)
    # Quick boundary check
    if np.any(np.abs(centers) > 0.5):
        return 1000.0  # large penalty
    return -_radius_from_centers(centers)


def _gradient_free_step(centers, step_size=0.01):
    """Simple repulsion-based local improvement."""
    n = len(centers)
    centers = centers.copy()
    for _ in range(50):
        forces = np.zeros_like(centers)
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.random.randn(3) * 0.001
                    dist = np.linalg.norm(diff)
                # Repulsive force
                force = diff / (dist * dist)
                forces[i] += force
                forces[j] -= force
            # Attract toward center to keep in bounds
            forces[i] -= 0.1 * centers[i]
        centers += step_size * forces
        # Project back into cube
        centers = np.clip(centers, -0.49, 0.49)
    return centers


def construct():
    start_time = time.time()
    
    # Try multiple initial configurations and local optimizations
    best_centers = None
    best_radius = 0.0
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE - TIME_MARGIN - elapsed()
    
    # Initial configuration 1: Grid-based
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init1 = grid[:N].copy()
    
    # Initial configuration 2: Random with jitter
    np.random.seed(42)
    init2 = np.random.uniform(-0.45, 0.45, size=(N, 3))
    
    # Initial configuration 3: Layered structure
    layers = []
    per_layer = 6  # 6*6=36, then 14 more in 2 layers
    for layer_idx in range(4):
        z_pos = (layer_idx + 0.5) / 4.0 - 0.5
        if layer_idx < 2:
            count = 15
        elif layer_idx < 4:
            count = 10
        else:
            count = 0
        for i in range(count):
            # Hexagonal-ish arrangement in layer
            row = i // 4
            col = i % 4
            x = (col + 0.5) / 4.0 - 0.5 + (0.125 if row % 2 else 0)
            y = (row + 0.5) / 4.0 - 0.5
            layers.append([x, y, z_pos])
    init3 = np.array(layers[:N])
    
    inits = [init1, init2, init3]
    
    for init_idx, init in enumerate(inits):
        if remaining() < 30.0:
            break
        
        # Local optimization using repulsion
        opt_centers = _gradient_free_step(init, step_size=0.02)
        r = _radius_from_centers(opt_centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = opt_centers.copy()
        
        # Try scipy optimization from this starting point
        if remaining() > 40.0:
            try:
                x0 = opt_centers.flatten()
                res = minimize(_neg_radius, x0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.success or res.fun < 0:
                    opt2 = res.x.reshape(-1, 3)
                    # Ensure within bounds
                    opt2 = np.clip(opt2, -0.49, 0.49)
                    r2 = _radius_from_centers(opt2)
                    if r2 > best_radius:
                        best_radius = r2
                        best_centers = opt2.copy()
            except Exception:
                pass
    
    # Final refinement with smaller steps
    if best_centers is not None and remaining() > 10.0:
        refined = _gradient_free_step(best_centers, step_size=0.005)
        r_ref = _radius_from_centers(refined)
        if r_ref > best_radius:
            best_radius = r_ref
            best_centers = refined
    
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
