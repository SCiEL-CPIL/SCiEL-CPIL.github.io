"""Rewritten program: gradient-based circle packing with multi-start and polish."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

_DEADLINE = 150.0  # seconds, with safety margin handled inside


def _compute_radius(centers):
    """Compute the maximum common radius for given centers in [-0.5, 0.5]^2."""
    # Boundary constraint: distance from center to nearest edge
    # For square [-0.5, 0.5]^2, the constraint is 0.5 - max(|x|, |y|)
    bounds = 0.5 - np.max(np.abs(centers), axis=1)
    min_bound = np.min(bounds)
    
    # Pairwise distances
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    return min(min_bound, min_pair)


def _hex_init(n):
    """Initialize n circles in a hexagonal lattice pattern within the square."""
    # Hexagonal lattice: rows offset by half spacing
    # We'll pack as many as possible in a hex pattern
    # spacing = 2*r, but we don't know r yet. Use a rough estimate.
    # For 200 circles in unit square, r ~ 0.036, so spacing ~ 0.072
    # Let's use a grid-like hex with spacing ~ 0.075
    
    centers = []
    spacing = 0.075
    dx = spacing
    dy = spacing * np.sqrt(3) / 2
    
    # Start from top-left corner area
    x_start = -0.5 + spacing / 2
    y_start = 0.5 - spacing / 2
    
    row = 0
    while len(centers) < n:
        offset = (row % 2) * (dx / 2)
        x = x_start + offset
        while x <= 0.5 - spacing / 2 and len(centers) < n:
            y = y_start - row * dy
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                centers.append([x, y])
            x += dx
        row += 1
    
    centers = np.array(centers[:n])
    return centers


def _random_init(n, seed=None):
    """Random initialization within the square."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-0.45, 0.45, size=(n, 2))
    return centers


def _optimize(centers, max_time=120.0):
    """Optimize circle packing using gradient ascent on radius.
    
    Uses a simple approach: iteratively push apart overlapping circles
    and pull inward from boundaries.
    """
    start_time = time.time()
    centers = centers.copy().astype(np.float64)
    
    # We'll use a simple iterative optimization:
    # 1. Compute forces from overlaps and boundary violations
    # 2. Move centers in direction of increasing min radius
    
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Use a cooling schedule for step size
    step_size = 0.001
    cooling = 0.999
    
    for iteration in range(1000000):
        elapsed = time.time() - start_time
        if elapsed > max_time - 5:  # safety margin
            break
        
        # Compute current radius
        bounds = 0.5 - np.max(np.abs(centers), axis=1)
        d = cdist(centers, centers)
        np.fill_diagonal(d, np.inf)
        
        current_r = min(np.min(bounds), np.min(d) / 2.0)
        
        if current_r > best_radius:
            best_radius = current_r
            best_centers = centers.copy()
        
        # Find the constraining pairs and boundaries
        # For each circle, compute the force that would increase its minimum constraint
        
        # Boundary forces: push away from nearest edge
        forces = np.zeros_like(centers)
        
        # For each circle, if it's close to a boundary, push it inward
        for i in range(N):
            cx, cy = centers[i]
            # Distance to each boundary
            dist_left = cx + 0.5
            dist_right = 0.5 - cx
            dist_bottom = cy + 0.5
            dist_top = 0.5 - cy
            
            min_dist = min(dist_left, dist_right, dist_bottom, dist_top)
            
            if min_dist < current_r * 1.5:  # Only apply force if close to boundary
                # Push toward center
                if min_dist == dist_left:
                    forces[i, 0] += 1.0
                elif min_dist == dist_right:
                    forces[i, 0] -= 1.0
                elif min_dist == dist_bottom:
                    forces[i, 1] += 1.0
                elif min_dist == dist_top:
                    forces[i, 1] -= 1.0
        
        # Pairwise repulsion for close pairs
        # Only consider pairs within some threshold
        threshold = current_r * 2.5
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < threshold and dist > 1e-10:
                    # Repulsive force
                    force_mag = (threshold - dist) / threshold
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
        
        # Normalize forces
        force_norm = np.linalg.norm(forces)
        if force_norm > 1e-10:
            forces /= force_norm
        
        # Update centers
        centers += step_size * forces
        
        # Clip to stay within bounds (with small margin)
        centers = np.clip(centers, -0.499, 0.499)
        
        # Cool down
        step_size *= cooling
        if step_size < 1e-7:
            break
    
    return best_centers, best_radius


def _polish(centers, max_time=30.0):
    """Final polish with smaller steps to refine the solution."""
    start_time = time.time()
    centers = centers.copy().astype(np.float64)
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    step_size = 0.0001
    cooling = 0.9995
    
    for iteration in range(500000):
        elapsed = time.time() - start_time
        if elapsed > max_time - 2:
            break
        
        bounds = 0.5 - np.max(np.abs(centers), axis=1)
        d = cdist(centers, centers)
        np.fill_diagonal(d, np.inf)
        
        current_r = min(np.min(bounds), np.min(d) / 2.0)
        
        if current_r > best_radius:
            best_radius = current_r
            best_centers = centers.copy()
        
        forces = np.zeros_like(centers)
        
        for i in range(N):
            cx, cy = centers[i]
            dist_left = cx + 0.5
            dist_right = 0.5 - cx
            dist_bottom = cy + 0.5
            dist_top = 0.5 - cy
            min_dist = min(dist_left, dist_right, dist_bottom, dist_top)
            
            if min_dist < current_r * 1.3:
                if min_dist == dist_left:
                    forces[i, 0] += 1.0
                elif min_dist == dist_right:
                    forces[i, 0] -= 1.0
                elif min_dist == dist_bottom:
                    forces[i, 1] += 1.0
                elif min_dist == dist_top:
                    forces[i, 1] -= 1.0
        
        threshold = current_r * 2.2
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < threshold and dist > 1e-10:
                    force_mag = (threshold - dist) / threshold
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
        
        force_norm = np.linalg.norm(forces)
        if force_norm > 1e-10:
            forces /= force_norm
        
        centers += step_size * forces
        centers = np.clip(centers, -0.499, 0.499)
        
        step_size *= cooling
        if step_size < 1e-8:
            break
    
    return best_centers, best_radius


def construct():
    """Construct centers for N=200 equal circles in unit square [-0.5, 0.5]^2."""
    global _DEADLINE
    start_time = time.time()
    
    # Multi-start: try hex init and random inits
    best_centers = None
    best_radius = -np.inf
    
    # Try hexagonal initialization
    hex_centers = _hex_init(N)
    opt_centers, opt_radius = _optimize(hex_centers, max_time=40.0)
    if opt_radius > best_radius:
        best_radius = opt_radius
        best_centers = opt_centers.copy()
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - 20:
        # Not enough time for more starts, just polish
        polished, polished_r = _polish(best_centers, max_time=10.0)
        if polished_r > best_radius:
            best_radius = polished_r
            best_centers = polished
        return best_centers
    
    # Try a few random initializations
    for seed in [42, 123, 777]:
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 25:
            break
        
        rand_centers = _random_init(N, seed=seed)
        opt_centers, opt_radius = _optimize(rand_centers, max_time=30.0)
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers.copy()
    
    # Final polish
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 5  # 5 second safety margin
    if remaining > 5 and best_centers is not None:
        polished, polished_r = _polish(best_centers, max_time=max(5.0, remaining))
        if polished_r > best_radius:
            best_radius = polished_r
            best_centers = polished
    
    # Ensure validity: clip to square
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
