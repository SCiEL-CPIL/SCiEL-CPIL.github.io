# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools
import random

def _precompute_indices(n):
    """Precompute triangle indices for efficiency."""
    return np.array(list(itertools.combinations(range(n), 3)))

def _compute_dets(points, idx):
    """Compute 2 * signed area (determinant) for all triangles."""
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return det

def _compute_hull_area(points):
    """Compute convex hull area safely."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
        
    A_hull = _compute_hull_area(points)
    if A_hull <= 1e-12:
        return 0.0
        
    idx = _precompute_indices(n)
    dets = _compute_dets(points, idx)
    A_min = 0.5 * np.min(np.abs(dets))
    
    return A_min / A_hull

def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    idx = _precompute_indices(n)
    
    # Initializations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Concentric polygons (1 center, 5 inner, 7 outer)
    center = np.array([[0, 0]])
    angles_inner = np.linspace(0, 2*np.pi, 5, endpoint=False)
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    inner_pts = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    outer_pts = np.column_stack([1.0 * np.cos(angles_outer), 1.0 * np.sin(angles_outer)])
    inits.append(np.vstack([center, inner_pts, outer_pts]))
    
    # 3. Random
    rng = np.random.default_rng(seed=42)
    inits.append(rng.random((n, 2)) * 2 - 1)
    
    # 4. Grid
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts = np.column_stack([gx.ravel(), gy.ravel()])[:13]
    pts += rng.random((13, 2)) * 0.05
    inits.append(pts)
    
    # 5. Hexagonal lattice snippet
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    # Normalize to fit in [-1, 1]
    hex_pts = (hex_pts - hex_pts.mean(axis=0)) / (np.abs(hex_pts).max() + 1e-9)
    # Take 13 points
    hex_pts = hex_pts[:13]
    inits.append(hex_pts)
    
    # 6. Fibonacci spiral
    angles_fib = np.arange(n) * 2 * np.pi * 0.6180339887
    radii_fib = np.sqrt(np.arange(1, n + 1) / n)
    inits.append(np.column_stack([radii_fib * np.cos(angles_fib), radii_fib * np.sin(angles_fib)]))
    
    # 7. "Almost Regular" with jitter
    angles_jit = np.linspace(0, 2*np.pi, n, endpoint=False) + rng.random(n) * 0.1
    radii_jit = 1.0 + rng.random(n) * 0.1
    inits.append(np.column_stack([radii_jit * np.cos(angles_jit), radii_jit * np.sin(angles_jit)]))
    
    # 8. Two concentric rings (6 inner, 7 outer)
    angles_6 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    angles_7 = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    pts_6 = np.column_stack([0.6 * np.cos(angles_6), 0.6 * np.sin(angles_6)])
    pts_7 = np.column_stack([1.0 * np.cos(angles_7), 1.0 * np.sin(angles_7)])
    inits.append(np.vstack([pts_6, pts_7]))
    
    # 9. Three concentric rings (1 center, 4 mid, 8 outer)
    angles_4 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    angles_8 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    pts_4 = np.column_stack([0.5 * np.cos(angles_4), 0.5 * np.sin(angles_4)])
    pts_8 = np.column_stack([1.0 * np.cos(angles_8), 1.0 * np.sin(angles_8)])
    inits.append(np.vstack([center, pts_4, pts_8]))
    
    # Simulated Annealing
    for i_init, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
        
        current_pts = init_pts.copy()
        current_score = compute_score(current_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_pts.copy()
        
        # Annealing parameters
        T = 0.1
        T_min = 0.0001
        cooling = 0.9995
        steps = 0
        
        while T > T_min and time.time() < deadline:
            steps += 1
            new_pts = current_pts.copy()
            
            # Random move type
            move_type = rng.integers(0, 4)
            
            if move_type == 0:
                # Displace a random point
                idx_pt = rng.integers(0, n)
                displacement = rng.normal(0, T, size=2)
                new_pts[idx_pt] += displacement
            elif move_type == 1:
                # Rotate a random point around the center
                idx_pt = rng.integers(0, n)
                angle = rng.normal(0, T)
                c, s = np.cos(angle), np.sin(angle)
                R = np.array([[c, -s], [s, c]])
                new_pts[idx_pt] = R @ new_pts[idx_pt]
            elif move_type == 2:
                # Scale a random point from the center
                idx_pt = rng.integers(0, n)
                factor = 1 + rng.normal(0, T * 0.5)
                new_pts[idx_pt] *= factor
            else:
                # Swap two points (if they are distinct)
                idx1, idx2 = rng.choice(n, size=2, replace=False)
                new_pts[idx1], new_pts[idx2] = new_pts[idx2].copy(), new_pts[idx1].copy()
            
            new_score = compute_score(new_pts)
            
            # Acceptance probability
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts.copy()
            else:
                delta = new_score - current_score
                if delta < 0:
                    prob = np.exp(delta / T)
                    if rng.random() < prob:
                        current_pts = new_pts
                        current_score = new_score
            
            T *= cooling
        
        # Local refinement with small steps
        if time.time() < deadline - 10:
            T_refine = 0.01
            for _ in range(200):
                if time.time() > deadline:
                    break
                new_pts = current_pts.copy()
                idx_pt = rng.integers(0, n)
                displacement = rng.normal(0, T_refine, size=2)
                new_pts[idx_pt] += displacement
                new_score = compute_score(new_pts)
                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_pts.copy()
                T_refine *= 0.98

    # Fallback
    if best_points is None or best_score < 0.01:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
