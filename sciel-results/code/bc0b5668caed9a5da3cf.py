"""Rewritten program for circle packing in perimeter-4 rectangle."""
import time
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize

N = 21
DEADLINE = 110.0  # safety margin below 120s wall-clock
_START = time.time()


def _elapsed():
    return time.time() - _START


def _compute_voronoi_inradii(centers, W, H):
    """Compute maximal inscribed radius for each circle using Voronoi cells.
    
    centers: (N, 2) array of circle centers
    W, H: rectangle dimensions
    Returns: (N,) array of radii
    """
    n = len(centers)
    if n < 3:
        return np.array([min(W, H) / 2] * n)
    
    # Add boundary points far away to make Voronoi bounded
    # We'll use a large bounding box and clip
    pad = max(W, H) * 2
    points = np.vstack([
        centers,
        # Add 4 corner points of an expanded bounding box to close the Voronoi
        [[-pad, -pad], [W + pad, -pad], [W + pad, H + pad], [-pad, H + pad]]
    ])
    
    try:
        vor = Voronoi(points)
    except Exception:
        # Fallback: just use half the minimum distance to other circles and walls
        radii = np.zeros(n)
        for i in range(n):
            d_wall = min(centers[i, 0], W - centers[i, 0], centers[i, 1], H - centers[i, 1])
            d_circ = np.inf
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    d_circ = min(d_circ, d / 2)
            radii[i] = min(d_wall, d_circ)
        return radii
    
    radii = np.zeros(n)
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if len(region) == 0 or -1 in region:
            # Unbounded region - use wall distance
            d_wall = min(centers[i, 0], W - centers[i, 0], centers[i, 1], H - centers[i, 1])
            radii[i] = d_wall
            continue
        
        # Find minimum distance from center to each edge of the Voronoi cell
        min_dist = np.inf
        for j in range(len(region)):
            v1 = vor.vertices[region[j]]
            v2 = vor.vertices[region[(j + 1) % len(region)]]
            # Distance from point to line segment
            d = _point_to_segment_dist(centers[i], v1, v2)
            min_dist = min(min_dist, d)
        
        # Also consider wall distances (Voronoi with boundary points should handle this,
        # but double-check)
        d_wall = min(centers[i, 0], W - centers[i, 0], centers[i, 1], H - centers[i, 1])
        radii[i] = min(min_dist, d_wall)
    
    return radii


def _point_to_segment_dist(p, a, b):
    """Distance from point p to line segment ab."""
    ab = b - a
    t = np.dot(p - a, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    closest = a + t * ab
    return np.linalg.norm(p - closest)


def _objective(params, W, H):
    """Objective: maximize sum of Voronoi inradii.
    
    params: (N, 2) flattened centers
    Returns: negative sum of radii (for minimization)
    """
    centers = params.reshape(N, 2)
    
    # Boundary penalty: keep centers inside rectangle
    boundary_violation = 0.0
    for i in range(N):
        x, y = centers[i]
        if x < 0: boundary_violation += x * x
        if x > W: boundary_violation += (x - W) * (x - W)
        if y < 0: boundary_violation += y * y
        if y > H: boundary_violation += (y - H) * (y - H)
    
    if boundary_violation > 1e-10:
        return 1e6 + boundary_violation
    
    radii = _compute_voronoi_inradii(centers, W, H)
    return -np.sum(radii)


def _optimize_for_aspect(W, H, initial_centers, max_iter=2000, tol=1e-10):
    """Run optimization for a given rectangle aspect ratio."""
    if _elapsed() > DEADLINE:
        return None
    
    x0 = initial_centers.flatten()
    
    # Try L-BFGS-B with bounds
    bounds = [(0, W)] * N + [(0, H)] * N
    
    try:
        res = minimize(
            _objective, x0, args=(W, H),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50}
        )
    except Exception:
        return None
    
    if res.success or res.fun < 0:
        centers = res.x.reshape(N, 2)
        radii = _compute_voronoi_inradii(centers, W, H)
        return centers, radii, -res.fun
    
    return None


def _grid_initial(W, H):
    """Create a grid-based initial configuration."""
    # Use a 5x5 grid and pick 21 best positions, or use 4x5=20 + 1
    # Let's use rows/cols to distribute 21 circles
    # 21 = 3*7 or 4*5+1 or 5*4+1
    # Try a few layouts
    best = None
    best_val = -np.inf
    
    for rows in [3, 4, 5, 6]:
        cols = int(np.ceil(N / rows))
        # Adjust: we need exactly N circles
        # Place circles in a grid
        centers = []
        for r in range(rows):
            for c in range(cols):
                if len(centers) >= N:
                    break
                x = (c + 0.5) * W / cols
                y = (r + 0.5) * H / rows
                centers.append([x, y])
        
        if len(centers) != N:
            continue
        
        centers = np.array(centers)
        radii = _compute_voronoi_inradii(centers, W, H)
        val = np.sum(radii)
        if val > best_val:
            best_val = val
            best = centers
    
    if best is None:
        # Fallback: 5x5 grid, take first 21
        side = 5
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) >= N:
                    break
                centers.append([(j + 0.5) * W / side, (i + 0.5) * H / side])
        best = np.array(centers)
    
    return best


def construct():
    global _START
    _START = time.time()
    
    best_result = None
    best_score = -np.inf
    
    # Try different aspect ratios: W + 2H = 2 (perimeter 4 means 2(W+H)=4 so W+H=2)
    # W in (0, 2), H = 2 - W
    # Symmetric, so W in (0, 1] covers all unique aspects
    # Try several aspect ratios
    aspect_ratios = np.linspace(0.3, 1.0, 15)
    
    for ar in aspect_ratios:
        if _elapsed() > DEADLINE:
            break
        
        W = ar
        H = 2.0 - W
        if H <= 0 or W <= 0:
            continue
        
        init = _grid_initial(W, H)
        
        # Run optimization with multiple random restarts
        for restart in range(3):
            if _elapsed() > DEADLINE:
                break
            
            if restart == 0:
                x_init = init
            else:
                # Perturb initial
                x_init = init + np.random.randn(N, 2) * 0.05
                # Clip to bounds
                x_init[:, 0] = np.clip(x_init[:, 0], 0.01, W - 0.01)
                x_init[:, 1] = np.clip(x_init[:, 1], 0.01, H - 0.01)
            
            result = _optimize_for_aspect(W, H, x_init, max_iter=1500)
            if result is not None:
                centers, radii, score = result
                if score > best_score:
                    best_score = score
                    best_result = (centers, radii, W, H)
        
        # Also try a second phase with more iterations on the best aspect found so far
        if _elapsed() > DEADLINE:
            break
    
    # Final refinement on the best aspect
    if best_result is not None:
        centers, radii, W, H = best_result
        if _elapsed() < DEADLINE - 5:
            result = _optimize_for_aspect(W, H, centers, max_iter=3000, tol=1e-12)
            if result is not None:
                c2, r2, s2 = result
                if s2 > best_score:
                    best_result = (c2, r2, W, H)
                    best_score = s2
    
    # If we haven't found anything, fall back to the seed
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    centers, radii, W, H = best_result
    
    # Final safety check: ensure no overlaps and all circles inside
    # Apply a small shrink to guarantee validity
    result = np.zeros((N, 3))
    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        # Shrink slightly to ensure disjoint interiors
        r = r * 0.999
        # Ensure circle is inside rectangle
        r = min(r, x, W - x, y, H - y)
        if r < 1e-6:
            r = 1e-6
        result[i] = [x, y, r]
    
    return result
