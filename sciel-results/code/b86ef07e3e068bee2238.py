"""Pack 200 equal circles in a unit square to maximize common radius."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import KDTree

N = 200
_TIME_LIMIT = 140  # seconds, leave margin before 150s kill


def _score(points):
    """Compute the radius for a set of points in [-0.5, 0.5]^2."""
    if not np.all(np.isfinite(points)):
        return 0.0
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 0.0
    if len(points) < 2:
        min_pair = float('inf')
    else:
        d = pdist(points)
        min_pair = np.min(d)
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    min_boundary = np.min(dist_boundary)
    return min(min_pair / 2.0, min_boundary)


def _initialize_packing():
    """Generate a high-quality initial hexagonal packing."""
    best_points = None
    best_score = 0.0
    
    # Try various row counts for hexagonal packing
    for rows in range(12, 22):
        cols = N // rows
        remainder = N % rows
        
        # Calculate spacing
        # We want to fit `cols` items in width 1.0, and `rows` items in height 1.0
        # Hexagonal: vertical spacing is s * sqrt(3) / 2
        # Horizontal spacing is s
        # Width used: (cols - 1) * s + s/2 (offset) + s/2 (center) ? No, let's just center it.
        # Let's estimate s based on width and height.
        s_w = 1.0 / cols
        s_h = 1.0 / ((rows - 1) * np.sqrt(3) / 2 + np.sqrt(3)/2) # Approx
        s = min(s_w, s_h) * 0.95 # Slight shrink to ensure fit
        
        points = []
        for j in range(rows):
            offset = (j % 2) * s / 2
            # Center the row horizontally
            row_width = (cols - 1) * s
            start_x = -row_width / 2 + offset
            for i in range(cols):
                x = start_x + i * s
                y = -0.5 + s * np.sqrt(3) / 4 + j * s * np.sqrt(3) / 2
                points.append([x, y])
                
        # Add remainder points to the first row or distribute?
        # Better: Just take first N from a larger grid or adjust.
        # Let's just generate a grid of size (rows+1) x (cols+1) and pick best N?
        # Simpler: Generate exactly rows x cols, if N < rows*cols, drop last ones.
        # If N > rows*cols, this loop doesn't work.
        # Let's stick to rows*cols >= N.
        
        if len(points) < N:
            continue
            
        points = np.array(points[:N])
        
        # Center the whole set
        points[:, 0] -= np.mean(points[:, 0])
        points[:, 1] -= np.mean(points[:, 1])
        
        # Scale to fit in [-0.5, 0.5]
        max_abs = np.max(np.abs(points))
        if max_abs > 0.5:
            points *= 0.5 / max_abs
            
        sc = _score(points)
        if sc > best_score:
            best_score = sc
            best_points = points.copy()
            
    if best_points is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
        
    return best_points


def _force_directed_refine(points, deadline, iterations=2000, lr=0.01):
    """Use force-directed repulsion with KDTree for fast global spreading."""
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for it in range(iterations):
        if time.time() > deadline:
            break
        
        # Build KDTree for fast neighbor queries
        tree = KDTree(pts)
        # Find neighbors within a reasonable distance
        # Radius should be related to current min distance or fixed
        neighbors = tree.query_ball_point(pts, r=0.2)
        
        forces = np.zeros_like(pts)
        
        for i in range(len(pts)):
            for j in neighbors[i]:
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    forces[i] += np.random.randn(2) * 0.01
                elif dist < 0.1:
                    f_mag = (0.1 - dist) / 0.1
                    forces[i] += f_mag * diff / dist
        
        # Boundary forces
        for i in range(len(pts)):
            for k in range(2):
                if pts[i, k] < -0.45:
                    forces[i, k] += 0.5 * (-0.45 - pts[i, k])
                elif pts[i, k] > 0.45:
                    forces[i, k] += 0.5 * (0.45 - pts[i, k])
        
        # Update positions
        pts += lr * forces
        pts = np.clip(pts, -0.5, 0.5)
        
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_pts = pts.copy()
        
        # Decay learning rate
        lr *= 0.999
    
    return best_pts, best_score


def _bottleneck_refine(points, deadline, max_iter=5000):
    """Targeted optimization focusing on the bottleneck constraint (closest pair or wall)."""
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Find the bottleneck: closest pair or closest to wall
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        min_pair_dist = np.min(D)
        i_idx, j_idx = np.unravel_index(np.argmin(D), D.shape)
        
        dist_boundary = 0.5 - np.max(np.abs(pts), axis=1)
        min_boundary = np.min(dist_boundary)
        wall_idx = np.argmin(dist_boundary)
        
        # Determine which is the bottleneck
        if min_pair_dist / 2.0 < min_boundary:
            # Closest pair is bottleneck
            diff = pts[i_idx] - pts[j_idx]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                pts[i_idx] += np.random.randn(2) * 0.001
                pts[j_idx] -= np.random.randn(2) * 0.001
            else:
                # Push apart
                push = 0.002 * diff / dist
                pts[i_idx] += push
                pts[j_idx] -= push
        else:
            # Wall is bottleneck
            # Move the point closest to wall inward
            for k in range(2):
                if pts[wall_idx, k] < -0.5:
                    pts[wall_idx, k] += 0.001
                elif pts[wall_idx, k] > 0.5:
                    pts[wall_idx, k] -= 0.001
                else:
                    # Move away from the nearest wall
                    if abs(pts[wall_idx, k] + 0.5) < abs(pts[wall_idx, k] - 0.5):
                        pts[wall_idx, k] += 0.001
                    else:
                        pts[wall_idx, k] -= 0.001
        
        # Also apply small random perturbations to escape local minima
        pts += np.random.randn(N, 2) * 0.0002
        
        # Clamp to valid region
        pts = np.clip(pts, -0.5, 0.5)
        
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_pts = pts.copy()
    
    return best_pts, best_score


def _optimize(points, deadline):
    """Combined force-directed + bottleneck optimization."""
    # Phase 1: Force-directed for global spreading
    fd_pts, fd_score = _force_directed_refine(points, deadline, iterations=1000, lr=0.01)
    
    # Phase 2: Bottleneck-targeted refinement
    opt_pts, opt_score = _bottleneck_refine(fd_pts, deadline, max_iter=5000)
    
    if opt_score > fd_score:
        return opt_pts, opt_score
    else:
        return fd_pts, fd_score





def construct():
    deadline = time.time() + _TIME_LIMIT
    
    # Phase 1: Initialize
    init_pts = _initialize_packing()
    best_points = init_pts.copy()
    best_score = _score(init_pts)
    
    # Phase 2: Optimize
    opt_pts, opt_score = _optimize(best_points, deadline)
    if opt_score > best_score:
        best_points = opt_pts
        best_score = opt_score
        
    # Phase 3: Final polish with more bottleneck refinement if time permits
    if time.time() < deadline - 2:
        bn_pts, bn_score = _bottleneck_refine(best_points, deadline, max_iter=2000)
        if bn_score > best_score:
            best_points = bn_pts
            best_score = bn_score
    
    return best_points
