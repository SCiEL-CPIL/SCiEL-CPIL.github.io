"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _min_triangle_area_fast(points):
    """Compute min area of any triangle formed by the n points.
    points: (n, 2) array. Returns scalar min area."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = points[j] - points[i]  # (2,)
            diff_k = points - points[i]  # (n, 2)
            cross = d[0] * diff_k[:, 1] - d[1] * diff_k[:, 0]  # (n,)
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            areas = 0.5 * np.abs(cross[mask])
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area


def _bary_to_cart(bc):
    """Convert barycentric (b,c) coordinates (n,2) to Cartesian (n,2).
    a = 1 - b - c
    x = b + 0.5*c
    y = c*sqrt(3)/2
    """
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * (np.sqrt(3) / 2)
    return np.column_stack([x, y])


def _clip_to_triangle(points):
    """Project points into the equilateral triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    # y bounds
    y_max = np.sqrt(3) / 2
    y = np.clip(y, 0, y_max)
    
    # x bounds based on y
    # Left edge: x >= y / sqrt(3)
    # Right edge: x <= 1 - y / sqrt(3)
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    
    x = np.clip(x, x_min, x_max)
    
    return np.column_stack([x, y])


def _penalty_bary(bc):
    """Calculate penalty for barycentric coordinates being outside the triangle."""
    b = bc[:, 0]
    c = bc[:, 1]
    # b >= 0, c >= 0, b+c <= 1
    p1 = np.maximum(0, -b)
    p2 = np.maximum(0, -c)
    p3 = np.maximum(0, b + c - 1)
    return np.sum(p1**2) + np.sum(p2**2) + np.sum(p3**2)


def _objective_bary_nelder(bc_flat, penalty_weight=100.0):
    """
    Objective function for Nelder-Mead.
    Minimizes the negative of the minimum triangle area plus penalty.
    Input: flat array of length 2*N, representing (b_i, c_i) for each point.
    """
    bc = bc_flat.reshape(N, 2)
    points = _bary_to_cart(bc)
    min_area = _min_triangle_area_fast(points)
    pen = _penalty_bary(bc)
    return -min_area + penalty_weight * pen


def _generate_inits(rng, num_inits):
    """Generate diverse initializations in barycentric coordinates."""
    inits = []
    # 1. Random uniform
    for _ in range(num_inits // 2):
        bc = np.zeros((N, 2))
        for k in range(N):
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1)
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            bc[k, 0] = u
            bc[k, 1] = v
        inits.append(bc.flatten())
        
    # 2. Grid-like structures
    # Center
    center = np.array([1/3, 1/3])
    # Vertices
    v1 = np.array([0, 0])
    v2 = np.array([1, 0])
    v3 = np.array([0, 1])
    
    # Mix of vertices and center and midpoints
    for _ in range(num_inits - len(inits)):
        bc = np.zeros((N, 2))
        # Pick random mix
        for k in range(N):
            r = rng.random()
            if r < 0.2:
                bc[k] = v1 + 0.1 * rng.normal(size=2)
            elif r < 0.4:
                bc[k] = v2 + 0.1 * rng.normal(size=2)
            elif r < 0.6:
                bc[k] = v3 + 0.1 * rng.normal(size=2)
            elif r < 0.8:
                bc[k] = center + 0.1 * rng.normal(size=2)
            else:
                u = rng.uniform(0, 1)
                v = rng.uniform(0, 1)
                if u + v > 1:
                    u = 1 - u
                    v = 1 - v
                bc[k] = [u, v]
        # Clip to valid range to ensure start is somewhat valid
        bc[:, 0] = np.clip(bc[:, 0], 0, 1)
        bc[:, 1] = np.clip(bc[:, 1], 0, 1)
        # Ensure b+c <= 1 by scaling if needed
        sums = bc[:, 0] + bc[:, 1]
        mask = sums > 1
        if np.any(mask):
            bc[mask] /= sums[mask, None]
        inits.append(bc.flatten())
        
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    rng = np.random.default_rng(42)
    best_points = None
    best_score = -float('inf')
    
    # Nelder-Mead options
    options = {
        'maxiter': 2000,
        'xatol': 1e-8,
        'fatol': 1e-8,
        'adaptive': True
    }
    
    n_starts = 30
    
    inits = _generate_inits(rng, n_starts)
    
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 5:
            break
            
        try:
            result = minimize(
                _objective_bary_nelder,
                x0,
                method='Nelder-Mead',
                options=options
            )
            
            bc_opt = result.x.reshape(N, 2)
            
            # Clean up: clip to valid barycentric region
            b = np.clip(bc_opt[:, 0], 0, 1)
            c = np.clip(bc_opt[:, 1], 0, 1)
            sums = b + c
            mask = sums > 1
            if np.any(mask):
                b[mask] /= sums[mask]
                c[mask] /= sums[mask]
            
            bc_opt = np.column_stack([b, c])
            points_opt = _bary_to_cart(bc_opt)
            
            # Final safety clip to Cartesian triangle just in case
            points_opt = _clip_to_triangle(points_opt)
            
            score = _min_triangle_area_fast(points_opt)
            
            if score > best_score:
                best_score = score
                best_points = points_opt.copy()
                
        except Exception:
            continue
    
    if best_points is None:
        # Fallback: Centered grid
        h = np.sqrt(3) / 2
        best_points = np.array([
            [0.5, 0.0], [0.0, 0.0], [1.0, 0.0],
            [0.25, h/2], [0.75, h/2], [0.5, h],
            [0.5, h/4], [0.25, h/4], [0.75, h/4],
            [0.5, 3*h/4], [0.5, 2*h/4]
        ])
        
    # Final safety check: ensure returned points are inside
    best_points = _clip_to_triangle(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
