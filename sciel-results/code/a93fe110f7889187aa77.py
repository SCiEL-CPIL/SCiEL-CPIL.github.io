"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Voronoi, ConvexHull
from scipy.spatial.distance import cdist

def _voronoi_radius(xy):
    """Compute the max radius for each circle given positions xy (n,2).
    Radius is limited by distance to walls and to Voronoi edges (nearest neighbors)."""
    n = len(xy)
    radii = np.empty(n)
    # Distance to walls
    wall_dist = np.minimum(np.minimum(xy[:, 0], 1.0 - xy[:, 0]),
                           np.minimum(xy[:, 1], 1.0 - xy[:, 1]))
    radii[:] = wall_dist
    if n < 3:
        return radii
    # Build Voronoi with boundary points
    # Add boundary points to capture wall constraints via Voronoi
    # Use a large bounding box of extra points
    margin = 2.0
    corners = np.array([[-margin, -margin], [1+margin, -margin],
                        [-margin, 1+margin], [1+margin, 1+margin]])
    # Add mid-edge points
    mids = np.array([[-margin, 0.5], [1+margin, 0.5],
                     [0.5, -margin], [0.5, 1+margin]])
    extra = np.vstack([corners, mids])
    all_pts = np.vstack([xy, extra])
    try:
        vor = Voronoi(all_pts)
        # For each original point, find the minimum distance to any Voronoi edge
        for i in range(n):
            region_id = vor.point_region[i]
            region = vor.regions[region_id]
            if -1 in region:
                # Unbounded region - already handled by wall_dist
                continue
            min_edge_dist = wall_dist[i]
            for edge_idx in region:
                if edge_idx < 0:
                    continue
                v1 = vor.vertices[edge_idx * 2]
                v2 = vor.vertices[edge_idx * 2 + 1]
                # Distance from xy[i] to line segment v1-v2
                # Project point onto segment
                d = v2 - v1
                seg_len2 = np.dot(d, d)
                if seg_len2 < 1e-15:
                    dist = np.linalg.norm(xy[i] - v1)
                else:
                    t = np.dot(xy[i] - v1, d) / seg_len2
                    t = max(0.0, min(1.0, t))
                    closest = v1 + t * d
                    dist = np.linalg.norm(xy[i] - closest)
                min_edge_dist = min(min_edge_dist, dist)
            radii[i] = min(radii[i], min_edge_dist)
    except Exception:
        pass
    return radii

def _hex_init(n):
    """Hexagonal close-packing initialization for n circles in unit square."""
    # Optimal hex packing: rows offset by half spacing
    # For n circles, find best arrangement
    # Try different row counts
    best_xy = None
    best_score = -1
    for rows in range(1, n + 1):
        cols = int(np.ceil(n / rows))
        # Hex spacing: horizontal = s, vertical = s * sqrt(3)/2
        # We want to fit in [0,1]^2
        # With cols circles in a row, horizontal span = (cols-1)*s
        # With rows, vertical span = (rows-1)*s*sqrt(3)/2
        # Maximize s subject to (cols-1)*s <= 1 and (rows-1)*s*sqrt(3)/2 <= 1
        if cols > 1:
            s_h = 1.0 / (cols - 1)
        else:
            s_h = 1.0
        if rows > 1:
            s_v = 1.0 / ((rows - 1) * np.sqrt(3) / 2)
        else:
            s_v = 1.0
        s = min(s_h, s_v)
        if s <= 0:
            continue
        # Generate positions
        pts = []
        for r in range(rows):
            for c in range(cols):
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
            if len(pts) >= n:
                break
            # Offset odd rows
            offset = (s / 2) if r % 2 == 1 else 0
            x = c * s + offset
            y = r * s * np.sqrt(3) / 2
            # Center in the square
            pts.append([x, y])
        if len(pts) < n:
            continue
        pts = np.array(pts[:n])
        # Center the arrangement
        # Compute bounding box
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)
        # Shift to center
        cx = (max_x - min_x) / 2
        cy = (max_y - min_y) / 2
        pts[:, 0] = 0.5 - cx + pts[:, 0] - min_x
        pts[:, 1] = 0.5 - cy + pts[:, 1] - min_y
        # Ensure within [0,1]
        pts[:, 0] = np.clip(pts[:, 0], 0, 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, 1)
        # Score: sum of Voronoi radii
        radii = _voronoi_radius(pts)
        score = np.sum(radii)
        if score > best_score:
            best_score = score
            best_xy = pts.copy()
    if best_xy is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
        best_xy = np.array(cells[:n])
    return best_xy

def _perturb_local_search(xy, radii, max_iters=2000, deadline=None):
    """Perturbation-based local search to improve sum of Voronoi radii."""
    n = len(xy)
    best_score = np.sum(radii)
    best_xy = xy.copy()
    best_radii = radii.copy()
    step = 0.02
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Pick a random circle to move
        i = rng.integers(n)
        # Try several perturbation sizes
        for _ in range(3):
            dx = rng.uniform(-step, step)
            dy = rng.uniform(-step, step)
            new_xy = best_xy.copy()
            new_xy[i] = [best_xy[i, 0] + dx, best_xy[i, 1] + dy]
            # Clamp to [0,1]
            new_xy[i, 0] = np.clip(new_xy[i, 0], 0, 1)
            new_xy[i, 1] = np.clip(new_xy[i, 1], 0, 1)
            new_radii = _voronoi_radius(new_xy)
            new_score = np.sum(new_radii)
            if new_score > best_score + 1e-10:
                best_score = new_score
                best_xy = new_xy.copy()
                best_radii = new_radii.copy()
        # Anneal step size
        step *= 0.999
        if step < 0.001:
            step = 0.001
    
    return best_xy, best_radii, best_score

def construct():
    deadline = time.time() + 145  # Safety margin
    
    # Step 1: Hexagonal initialization
    xy = _hex_init(N)
    radii = _voronoi_radius(xy)
    score = np.sum(radii)
    
    # Step 2: Local search with perturbation
    xy, radii, score = _perturb_local_search(xy, radii, max_iters=5000, deadline=deadline)
    
    # Step 3: Try multiple restarts if time permits
    rng = np.random.default_rng(123)
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        # Perturb initial solution
        xy_r = xy + rng.uniform(-0.05, 0.05, size=(N, 2))
        xy_r = np.clip(xy_r, 0, 1)
        radii_r = _voronoi_radius(xy_r)
        score_r = np.sum(radii_r)
        if score_r > score:
            xy, radii, score = _perturb_local_search(xy_r, radii_r, max_iters=3000, deadline=deadline)
        else:
            # Still try local search from this perturbation
            xy_r2, radii_r2, score_r2 = _perturb_local_search(xy_r, radii_r, max_iters=2000, deadline=deadline)
            if score_r2 > score:
                xy, radii, score = xy_r2, radii_r2, score_r2
    
    # Final: ensure validity
    result = np.column_stack([xy, radii])
    # Clamp radii to be non-negative and ensure circles stay in [0,1]
    result[:, 0] = np.clip(result[:, 0], 0, 1)
    result[:, 1] = np.clip(result[:, 1], 0, 1)
    result[:, 2] = np.clip(result[:, 2], 0, 0.5)
    # Re-check: for each circle, radius must not exceed distance to walls
    for i in range(N):
        max_r = min(result[i, 0], 1 - result[i, 0], result[i, 1], 1 - result[i, 1])
        result[i, 2] = min(result[i, 2], max_r)
    
    return result
# EVOLVE-BLOCK-END

