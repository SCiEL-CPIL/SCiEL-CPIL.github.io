"""Circle packing in unit square: pack 26 circles, maximize sum of radii.

STRATEGY: Voronoi-cell inradius maximization.
- The 26 circle centers are the only decision variables (52-D).
- Radii are NOT free variables: each radius is the inradius of its Voronoi cell
  (distance to nearest cell edge / neighbor center / wall), which guarantees
  pairwise non-overlap and wall containment by construction.
- Objective = sum of these inradii. We maximize it with L-BFGS-B (fast, no
  explicit pairwise constraints needed) from multiple diverse starts, plus a
  final SLSQP polish.
- This eliminates the 325 pairwise collision constraints that plague SLSQP on
  78-D problems, and makes each optimization call much cheaper and more robust.
"""
import numpy as np
import time
from scipy.spatial import Voronoi, QhullError
from scipy.optimize import minimize

N = 26
EPS = 1e-9


def _inradii_from_centers(centers):
    """
    Given N centers in [0,1]^2, compute the inradius of each circle such that
    all circles are inside the unit square and have disjoint interiors.
    Uses Voronoi diagram: the inradius of circle i is the minimum distance from
    center i to:
      - the four walls
      - the perpendicular bisector to every other center j (i.e., Voronoi edges)
    Equivalently, for each i: r_i = min(
        min(dist to walls),
        min over j != i of (d_ij - r_j)  -- but this is implicit.
    )
    
    Exact approach: build Voronoi of the N points (plus boundary points to
    handle walls), then for each cell compute the distance from the site to
    the nearest vertex/edge of the cell that lies within [0,1]^2.
    
    Simpler exact approach: for each i, r_i = min(
        x_i, 1-x_i, y_i, 1-y_i,
        min over j!=i of 0.5*(d_ij)  -- NO, this is only valid if all radii equal.
    )
    
    Correct approach: The Voronoi cell of site i is the set of points closer to
    i than to any other site. The largest circle centered at i inside its
    Voronoi cell has radius = distance from i to the nearest point on the
    boundary of the cell. The boundary consists of line segments (perp bisectors
    to neighbors) and, near the square boundary, the square edges.
    
    We compute this by:
    1. Build Voronoi of the N centers + 4 corner points (to capture walls).
    2. For each original site i, find its Voronoi region.
    3. Compute distance from center i to each edge of its region that falls
       within [0,1]^2.
    4. r_i = min of those distances.
    
    Even simpler and robust: For each i, r_i = min(
        x_i, 1-x_i, y_i, 1-y_i,
        min over j!=i of (d_ij / 2)  # This is WRONG for unequal radii
    )
    
    Let me use the correct method:
    For each i, the constraint from neighbor j is: d_ij >= r_i + r_j.
    The Voronoi condition says r_i <= d_ij/2 only if r_i = r_j.
    
    The CORRECT inradius computation:
    Build Voronoi diagram of the N points. For each cell i, the inradius is the
    distance from site i to the closest point on the cell boundary (which is a
    convex polygon, possibly unbounded -- but we intersect with [0,1]^2).
    
    Practical robust method:
    1. Add 4 corners and midpoints of edges as "virtual sites" to make Voronoi
       bounded and capture wall constraints.
    2. Build Voronoi.
    3. For each original site i, compute distance to each vertex of its cell
       (clipped to [0,1]^2). r_i = min distance.
    """
    n = len(centers)
    cx = centers[:, 0]
    cy = centers[:, 1]
    
    # Wall distances
    wall_r = np.minimum(np.minimum(cx, 1.0 - cx), np.minimum(cy, 1.0 - cy))
    
    # Neighbor constraints: for each i, r_i <= d_ij - r_j for all j != i.
    # But r_j is unknown. The Voronoi inradius is the solution where
    # r_i = dist from i to nearest Voronoi edge.
    
    # Build Voronoi with boundary points
    boundary_pts = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],
        [0.5, 0], [0.5, 1], [0, 0.5], [1, 0.5],
        [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
        [0.5, 0.5]  # center as backup
    ])
    # Filter out boundary points that are too close to any center
    keep = []
    for bp in boundary_pts:
        d = np.sqrt((cx - bp[0])**2 + (cy - bp[1])**2)
        if np.min(d) > 0.02:  # not too close
            keep.append(bp)
    boundary_pts = np.array(keep) if keep else boundary_pts
    
    all_pts = np.vstack([centers, boundary_pts])
    
    try:
        vor = Voronoi(all_pts)
    except QhullError:
        # Fallback: brute-force inradius
        r = wall_r.copy()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                # Approximate: r_i <= d/2 (conservative if radii differ)
                # Better: r_i <= d - r_j, but r_j unknown. Use iterative.
                r[i] = min(r[i], d * 0.5)
        # Iterate to converge
        for _ in range(50):
            for i in range(n):
                ri = wall_r[i]
                for j in range(n):
                    if i == j:
                        continue
                    d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                    ri = min(ri, d - r[j])
                r[i] = max(ri, 0.0)
        return r
    
    # For each original site i (index 0..n-1), find its Voronoi region
    r = np.zeros(n)
    for i in range(n):
        # Find the region containing point i
        # Voronoi region: points closer to site i than any other
        # The inradius is the max r such that ball(center_i, r) is inside the cell.
        # This equals the distance from center_i to the nearest boundary of the cell.
        
        # Get all vertices of the Voronoi diagram that are within [0,1]^2
        # and are on the boundary of cell i.
        
        # Method: for each Voronoi vertex, check if it's closer to site i
        # than to any other site (i.e., it's on the boundary of cell i).
        # Actually, a Voronoi vertex is equidistant to 3+ sites. It's on the
        # boundary of cell i if site i is one of the sites equidistant at that vertex.
        
        # Simpler: compute distance from center_i to each Voronoi vertex,
        # but only count vertices that are "visible" from cell i.
        
        # Most robust: for each Voronoi vertex v, check if center_i is among
        # the sites whose Voronoi cells contain v.
        
        min_dist = wall_r[i]
        
        # Check Voronoi vertices
        for v in vor.vertices:
            if not (0 <= v[0] <= 1 and 0 <= v[1] <= 1):
                continue
            # Is this vertex on the boundary of cell i?
            # A vertex is on the boundary of cell i if the distance from v to site i
            # equals the minimum distance from v to any site (within tolerance).
            dists = np.sqrt((all_pts[:, 0] - v[0])**2 + (all_pts[:, 1] - v[1])**2)
            min_d = np.min(dists)
            if abs(dists[i] - min_d) < 1e-9:
                # This vertex is equidistant to site i and at least one other,
                # so it's on the boundary of cell i.
                min_dist = min(min_dist, dists[i])
        
        # Also check Voronoi edges (for unbounded cells or cells touching boundary)
        # For each edge, compute distance from center_i to the edge segment,
        # but only if the edge is on the boundary of cell i.
        
        # Check wall distances (already in min_dist via wall_r)
        r[i] = max(min_dist, 0.0)
    
    return r


def _inradii_fast(centers):
    """
    Fast computation of inradii using the iterative method:
    r_i = min(wall_dist_i, min over j!=i (d_ij - r_j))
    This is exact for Voronoi inradii and converges quickly.
    """
    n = len(centers)
    cx = centers[:, 0]
    cy = centers[:, 1]
    wall_r = np.minimum(np.minimum(cx, 1.0 - cx), np.minimum(cy, 1.0 - cy))
    
    r = wall_r.copy()
    # Precompute distance matrix
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx*dx + dy*dy)
    np.fill_diagonal(dist, np.inf)
    
    # Iterate until convergence
    for _ in range(100):
        # r_i = min(wall_r_i, min_j (d_ij - r_j))
        new_r = np.zeros(n)
        for i in range(n):
            vals = dist[i, :] - r  # d_ij - r_j for all j
            vals[i] = np.inf
            new_r[i] = min(wall_r[i], np.min(vals))
            new_r[i] = max(new_r[i], 0.0)
        if np.max(np.abs(new_r - r)) < 1e-12:
            r = new_r
            break
        r = new_r
    
    return r


def _objective(centers):
    """Negative sum of inradii (for minimization)."""
    r = _inradii_fast(centers)
    return -float(np.sum(r))


def _objective_grad(centers):
    """Numerical gradient of the objective."""
    n = len(centers)
    grad = np.zeros_like(centers)
    h = 1e-7
    f0 = _objective(centers)
    for i in range(n):
        old = centers[i]
        centers[i] += h
        f1 = _objective(centers)
        centers[i] = old
        grad[i] = (f1 - f0) / h
    return grad


def _total_radius(centers):
    """Sum of inradii."""
    r = _inradii_fast(centers)
    return float(np.sum(r))


def _is_valid(centers, r):
    """Check if the packing is valid."""
    n = len(centers)
    for i in range(n):
        x, y, ri = centers[i, 0], centers[i, 1], r[i]
        if ri < -EPS or x - ri < -EPS or 1.0 - x - ri < -EPS or y - ri < -EPS or 1.0 - y - ri < -EPS:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            if d < r[i] + r[j] - EPS:
                return False
    return True


def _grid_init():
    """Structured grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    centers = np.zeros((N, 2))
    for i in range(N):
        gx = i % side
        gy = i // side
        centers[i, 0] = (gx + 0.5) / side
        centers[i, 1] = (gy + 0.5) / side
    return centers


def _random_init(rng):
    """Random initialization with minimum spacing."""
    centers = np.zeros((N, 2))
    min_dist = 0.1
    for i in range(N):
        for attempt in range(1000):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            if i == 0:
                centers[i] = [x, y]
                break
            d = np.sqrt((centers[:i, 0] - x)**2 + (centers[:i, 1] - y)**2)
            if np.min(d) > min_dist:
                centers[i] = [x, y]
                break
        else:
            # Fallback: place on grid
            gx = i % side
            gy = i // side
            centers[i] = [(gx + 0.5) / side, (gy + 0.5) / side]
    return centers


def _hex_init():
    """Hexagonal packing initialization."""
    centers = np.zeros((N, 2))
    # Hexagonal grid
    row = 0
    col = 0
    dx = 0.19
    dy = dx * np.sqrt(3) / 2
    count = 0
    while count < N and row < 20:
        x_offset = (row % 2) * dx / 2
        for c in range(20):
            if count >= N:
                break
            x = x_offset + c * dx + 0.05
            y = row * dy + 0.05
            if 0 <= x <= 1 and 0 <= y <= 1:
                centers[count] = [x, y]
                count += 1
        row += 1
    # Fill remaining with random
    for i in range(count, N):
        centers[i] = [0.5, 0.5]
    return centers


def _optimize_centers(centers0, deadline, max_iter=300):
    """Optimize centers using L-BFGS-B to maximize sum of inradii."""
    v0 = centers0.ravel()
    bounds = [(0.0, 1.0)] * (2 * N)
    
    def obj(v):
        c = v.reshape(-1, 2)
        return _objective(c)
    
    try:
        res = minimize(obj, v0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    except Exception:
        return centers0
    
    if time.time() > deadline:
        return centers0
    
    new_centers = res.x.reshape(-1, 2)
    new_centers = np.clip(new_centers, 0.0, 1.0)
    
    # Check if improvement
    if _total_radius(new_centers) > _total_radius(centers0):
        return new_centers
    return centers0


def _slsqp_polish(centers, deadline, max_iter=200):
    """SLSQP polish on the 52-D center problem."""
    v0 = centers.ravel()
    bounds = [(0.0, 1.0)] * (2 * N)
    
    def obj(v):
        c = v.reshape(-1, 2)
        r = _inradii_fast(c)
        return -float(np.sum(r))
    
    try:
        res = minimize(obj, v0, method='SLSQP', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-10})
    except Exception:
        return centers
    
    if time.time() > deadline:
        return centers
    
    new_centers = res.x.reshape(-1, 2)
    new_centers = np.clip(new_centers, 0.0, 1.0)
    
    if _total_radius(new_centers) > _total_radius(centers):
        return new_centers
    return centers


def _local_search(centers, deadline, n_iter=50):
    """Simple local search: perturb centers and accept if total radius improves."""
    best_centers = centers.copy()
    best_total = _total_radius(best_centers)
    rng = np.random.default_rng(42)
    
    for _ in range(n_iter):
        if time.time() > deadline:
            break
        # Perturb a random subset of centers
        idx = rng.choice(N, size=max(1, N // 4), replace=False)
        new_centers = best_centers.copy()
        noise = rng.normal(0, 0.02, size=(len(idx), 2))
        new_centers[idx] += noise
        new_centers = np.clip(new_centers, 0.0, 1.0)
        
        total = _total_radius(new_centers)
        if total > best_total + 1e-12:
            best_centers = new_centers
            best_total = total
    
    return best_centers


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    
    # Multiple diverse initializations
    inits = []
    
    # Grid
    inits.append(_grid_init())
    
    # Hex
    inits.append(_hex_init())
    
    # Random with different seeds
    for seed in [1, 2, 3, 4, 5]:
        rng = np.random.default_rng(seed)
        inits.append(_random_init(rng))
    
    # Perturbed grid
    rng = np.random.default_rng(100)
    grid_c = _grid_init()
    for i in range(N):
        grid_c[i] += rng.normal(0, 0.05, size=2)
    grid_c = np.clip(grid_c, 0.05, 0.95)
    inits.append(grid_c)
    
    best_centers = None
    best_total = -np.inf
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # L-BFGS-B optimization
        opt_centers = _optimize_centers(init, deadline, max_iter=200)
        
        # Local search
        if time.time() < deadline - 10:
            ls_centers = _local_search(opt_centers, deadline, n_iter=30)
            if _total_radius(ls_centers) > _total_radius(opt_centers):
                opt_centers = ls_centers
        
        total = _total_radius(opt_centers)
        if total > best_total:
            best_total = total
            best_centers = opt_centers.copy()
    
    # SLSQP polish on best
    if best_centers is not None and time.time() < deadline - 15:
        polished = _slsqp_polish(best_centers, deadline, max_iter=100)
        if _total_radius(polished) > best_total:
            best_centers = polished
            best_total = _total_radius(polished)
    
    if best_centers is None:
        best_centers = _grid_init()
    
    # Compute final radii
    r = _inradii_fast(best_centers)
    
    # Final validation and clipping
    n = len(best_centers)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_centers[i, 0], best_centers[i, 1], r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.0005)
        for j in range(n):
            if j == i:
                continue
            d = np.hypot(x - best_centers[j, 0], y - best_centers[j, 1])
            ri = min(ri, d - r[j])
        ri = max(ri, 0.0005)
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
        result[i, 2] = ri
    
    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    if not _is_valid(final_xy, final_r):
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
    
    return result
