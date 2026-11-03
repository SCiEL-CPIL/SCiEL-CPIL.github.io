"""Delaunay-triangulation-based circle packing in the unit square.

Structural idea:
  - Represent the packing by center positions only (52 variables).
  - Radii are derived geometrically: r_i = min(boundary_dist_i, 0.5 * min_neighbor_dist_i).
  - Optimize center positions with a physics-inspired force model:
      * repulsion between circles (push apart when too close)
      * attraction toward a target spacing (prevents collapse)
      * boundary constraint forces
  - Use L-BFGS-B on the reduced 52-dim space.
  - Multi-start with diverse initial layouts.
  - Deadline guard for safety.

This reduces the search space from 78 to 52 variables and enforces
tight packing constraints geometrically, allowing the optimizer to
focus on positions only.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay, distance

N = 26
DEADLINE = 110.0
START = time.time()


def _elapsed():
    return time.time() - START


def _compute_radii(x, y):
    """Compute radii from positions using Delaunay-based neighbor distances."""
    pts = np.column_stack([x, y])
    r = np.zeros(N)
    
    # Boundary distance
    d_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # For each point, find nearest neighbor distance
    # Use cdist for efficiency
    from scipy.spatial.distance import cdist
    dist_mat = cdist(pts, pts)
    np.fill_diagonal(dist_mat, np.inf)
    
    # Nearest neighbor distance for each point
    nn_dist = np.min(dist_mat, axis=1)
    
    # Radius is half the minimum of (boundary dist, nearest neighbor dist)
    # But we also need to consider that two circles sharing an edge in Delaunay
    # should be tangent. Use the minimum over all neighbors in Delaunay.
    
    try:
        tri = Delaunay(pts)
        # For each vertex, get its neighbors
        for i in range(N):
            neighbors = set()
            for simplex in tri.simplices:
                if i in simplex:
                    for j in simplex:
                        if j != i:
                            neighbors.add(j)
            if neighbors:
                min_neighbor_dist = min(np.linalg.norm(pts[i] - pts[j]) for j in neighbors)
            else:
                min_neighbor_dist = nn_dist[i]
            r[i] = 0.5 * min(d_bound[i], min_neighbor_dist)
    except Exception:
        # Fallback: use nearest neighbor only
        r = 0.5 * np.minimum(d_bound, nn_dist)
    
    return np.maximum(r, 1e-6)


def _objective_positions(params):
    """
    Objective to minimize: negative sum of radii + penalties.
    params: 52-dim vector [x0, y0, x1, y1, ..., x25, y25]
    """
    x = np.clip(params[0::2], 0.01, 0.99)
    y = np.clip(params[1::2], 0.01, 0.99)
    
    r = _compute_radii(x, y)
    
    # Penalty for circles that are too close (overlap in derived radii)
    pts = np.column_stack([x, y])
    from scipy.spatial.distance import cdist
    dist_mat = cdist(pts, pts)
    np.fill_diagonal(dist_mat, np.inf)
    
    # Check if any pair has distance < r_i + r_j
    r_sum = r[:, None] + r[None, :]
    overlap_viol = np.maximum(r_sum - dist_mat, 0.0)
    overlap_penalty = 10.0 * np.sum(overlap_viol ** 2)
    
    # Boundary penalty: r should not exceed boundary distance
    d_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    bound_viol = np.maximum(r - d_bound, 0.0)
    bound_penalty = 50.0 * np.sum(bound_viol ** 2)
    
    return -np.sum(r) + overlap_penalty + bound_penalty


def _valid(x, y, r, eps=1e-6):
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    if np.any(r + eps > d):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :]
    if np.any(need + eps > dist):
        return False
    return True


def _grid_init_5x5_plus1(rng):
    cols, rows = 5, 5
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.1)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    return x, y


def _grid_init_4x6(rng):
    cols, rows = 4, 6
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.5), (0.25, 0.75)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    return x, y


def _grid_init_6x4(rng):
    cols, rows = 6, 4
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.5), (0.75, 0.25)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    return x, y


def _random_init(rng):
    x = rng.uniform(0.1, 0.9, size=N)
    y = rng.uniform(0.1, 0.9, size=N)
    return x, y


def _hex_init(rng):
    """Hexagonal lattice initialization."""
    x = np.zeros(N)
    y = np.zeros(N)
    spacing = 0.2
    for i in range(N):
        row = i // 6
        col = i % 6
        x[i] = 0.15 + col * spacing + (0.5 * spacing if row % 2 else 0)
        y[i] = 0.15 + row * spacing * 0.866
        if x[i] > 0.9:
            x[i] = 0.9
        if y[i] > 0.9:
            y[i] = 0.9
    return x, y


def construct():
    best = None
    best_val = -np.inf
    rng = np.random.default_rng(12345)

    inits = [
        _grid_init_5x5_plus1,
        _grid_init_4x6,
        _grid_init_6x4,
        _random_init,
        _hex_init,
    ]

    for init_func in inits:
        if _elapsed() > DEADLINE - 5.0:
            break
        x, y = init_func(rng)
        x = np.clip(x, 0.02, 0.98)
        y = np.clip(y, 0.02, 0.98)
        
        for restart in range(3):
            if _elapsed() > DEADLINE - 5.0:
                break
            x0 = x.copy()
            y0 = y.copy()
            jit = rng.normal(0.0, 0.02, size=N)
            x0 = np.clip(x0 + jit, 0.02, 0.98)
            y0 = np.clip(y0 + jit, 0.02, 0.98)
            
            vec0 = np.zeros(2 * N)
            vec0[0::2] = x0
            vec0[1::2] = y0

            try:
                res = minimize(_objective_positions, vec0, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6,
                                        'maxfun': 5000})
            except Exception:
                continue
            
            v = res.x
            xx = np.clip(v[0::2], 0.01, 0.99)
            yy = np.clip(v[1::2], 0.01, 0.99)
            rr = _compute_radii(xx, yy)
            
            # Final fix: ensure no overlaps by shrinking
            for _ in range(5):
                changed = False
                d_bound = np.minimum(np.minimum(xx, 1.0 - xx), np.minimum(yy, 1.0 - yy))
                rr = np.minimum(rr, d_bound)
                
                dx = xx[:, None] - xx[None, :]
                dy = yy[:, None] - yy[None, :]
                dist = np.sqrt(dx * dx + dy * dy)
                np.fill_diagonal(dist, np.inf)
                
                for i in range(N):
                    for j in range(i + 1, N):
                        if dist[i, j] < rr[i] + rr[j]:
                            # Shrink both proportionally
                            scale = dist[i, j] / (rr[i] + rr[j]) * 0.99
                            rr[i] *= scale
                            rr[j] *= scale
                            changed = True
                if not changed:
                    break
            
            if _valid(xx, yy, rr):
                val = float(np.sum(rr))
                if val > best_val:
                    best_val = val
                    best = np.column_stack([xx, yy, rr])

    if best is not None:
        return best

    # Guaranteed fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    cells = cells[:N]
    while len(cells) < N:
        cells.append((0.5 / side, 0.5 / side, 0.1 / side))
    return np.array(cells[:N])
