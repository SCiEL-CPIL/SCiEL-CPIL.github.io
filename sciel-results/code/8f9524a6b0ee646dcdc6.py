"""Pack 200 equal circles in a unit square to maximize common radius.

Uses a Delaunay triangulation guided edge-flip / combinatorial optimization
combined with gradient-based local refinement to escape local minima.
"""
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

N = 200
_START_TIME = time.time()
_TIME_LIMIT = 135  # seconds, leave margin before 150s kill


def _compute_radius(pts):
    """Compute the achievable radius for a set of points inside [-0.5, 0.5]^2."""
    if len(pts) < N:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    if np.any(~np.isfinite(pts)):
        return 0.0
    # Check inside square (strictly inside for valid packing)
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return 0.0
    # Min distance to wall
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    # Min pairwise distance
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    r = min(wall, min_pair / 2.0)
    return max(r, 0.0)


def _generate_hexagonal_grid(n):
    """Generate a hexagonal (close-packing) grid arrangement."""
    # For hexagonal packing, rows are offset
    # Try to fit n points in a hexagonal pattern
    pts_list = []
    row = 0
    while len(pts_list) < n:
        # Each row has alternating offsets
        row_width = 1.0 / (0.866 * (1 + 0.5 * (row % 2) * 0))  # approximate
        # Better: use spacing
        spacing_x = 1.0 / (np.ceil(np.sqrt(n / 1.5)) + 1)
        spacing_y = spacing_x * np.sqrt(3) / 2
        
        x_start = -0.5 + spacing_x / 2
        if row % 2 == 1:
            x_start += spacing_x / 2
        
        y = -0.5 + spacing_y / 2 + row * spacing_y
        
        col = 0
        while True:
            x = x_start + col * spacing_x
            if x > 0.5 - spacing_x / 2:
                break
            if y > 0.5 - spacing_y / 2:
                break
            pts_list.append([x, y])
            col += 1
        row += 1
        if row > 50:  # safety
            break
    
    pts = np.array(pts_list[:n], dtype=np.float64)
    if len(pts) < n:
        # Fill remaining with random points
        while len(pts) < n:
            pts = np.vstack([pts, np.random.uniform(-0.45, 0.45, 2)])
    return pts


def _generate_grid(pts_shape):
    """Generate a regular grid with given (rows, cols)."""
    rows, cols = pts_shape
    if rows * cols < N:
        rows = max(rows, N // cols + 1)
    dx = 1.0 / cols
    dy = 1.0 / rows
    xs = -0.5 + (np.arange(cols) + 0.5) * dx
    ys = -0.5 + (np.arange(rows) + 0.5) * dy
    grid = np.array([(x, y) for y in ys for x in xs])
    return grid[:N]


def _gradient_refine(pts, max_iters=5000, lr_init=0.01, lr_min=0.0001):
    """
    Gradient-based optimization of the minimum distance.
    Uses a smooth approximation of the min function.
    """
    pts = pts.copy().astype(np.float64)
    best_r = _compute_radius(pts)
    
    # We optimize: maximize min over all constraints
    # Use a soft-min approximation for gradient
    beta = 20.0  # temperature parameter for soft-min
    
    def _soft_min(values, beta):
        """Soft minimum of array of values."""
        # -soft_max(-values)
        neg_vals = -values
        max_neg = np.max(neg_vals)
        if max_neg > 0:
            exps = np.exp(neg_vals - max_neg)
        else:
            exps = np.exp(neg_vals)
        soft_max = max_neg + np.log(np.sum(exps))
        return -soft_max
    
    def _objective_and_grad(pts):
        """Compute negative soft-min radius and its gradient."""
        pts = np.asarray(pts, dtype=np.float64)
        n = len(pts)
        
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(dist_sq, 1e10)
        dist = np.sqrt(dist_sq)
        
        # Pairwise distances (lower triangle)
        iu = np.triu_indices(n, 1)
        pair_dists = dist[iu] / 2.0  # radius constraint
        
        # Wall distances
        wall_dists = np.concatenate([
            pts[:, 0] + 0.5,
            0.5 - pts[:, 0],
            pts[:, 1] + 0.5,
            0.5 - pts[:, 1],
        ])
        
        # All constraints
        all_constraints = np.concatenate([pair_dists, wall_dists])
        
        # Soft min
        r_soft = _soft_min(all_constraints, beta)
        
        # Gradient computation
        # d(soft_min)/dx = -sum(exp(c_i - max_c) * dc_i/dx) / sum(exp(c_i - max_c))
        # where c_i are the constraints
        
        max_c = np.max(all_constraints)
        exps = np.exp(all_constraints - max_c)
        exps_sum = np.sum(exps)
        
        # Gradient of each constraint w.r.t. pts
        grad = np.zeros_like(pts)
        
        # Pairwise distance gradients
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            w = exps[idx] / exps_sum
            # d(pair_dist[i,j]/2)/d(pts[i]) = -diff[i,j] / (2 * dist[i,j])
            # where diff[i,j] = pts[i] - pts[j]
            diff_ij = pts[i] - pts[j]
            d = dist[i, j]
            if d > 1e-12:
                grad[i] += w * (-diff_ij / (2 * d)) * 2  # factor of 2 from chain rule? Let me think...
                grad[j] += w * (diff_ij / (2 * d)) * 2
            # Actually: d(||pi-pj||/2)/dpi = -(pi-pj) / (2*||pi-pj||)
            # But we have exp weight... let me redo this more carefully
            
        # Let me use a cleaner approach: compute gradient numerically-like
        # Actually, let's use the analytical gradient properly
        
        # Recompute gradient properly
        grad = np.zeros_like(pts)
        
        # Pairwise
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            w = exps[idx] / exps_sum
            diff_ij = pts[i] - pts[j]
            d = np.linalg.norm(diff_ij)
            if d > 1e-12:
                # d(c_ij)/d(pts[i]) where c_ij = ||pi-pj||/2
                # = (pi-pj)/(2*||pi-pj||)
                grad[i] += w * diff_ij / (2 * d)
                grad[j] += w * (-diff_ij / (2 * d))
        
        # Wall distances
        # c_{2n} = pts[:,0]+0.5, grad = [1,0]
        # c_{2n+1} = 0.5-pts[:,0], grad = [-1,0]
        # c_{2n+2} = pts[:,1]+0.5, grad = [0,1]
        # c_{2n+3} = 0.5-pts[:,1], grad = [0,-1]
        for k in range(n):
            grad[k] += exps[2*n + k] / exps_sum * np.array([1.0, 0.0])
            grad[k] += exps[2*n + n + k] / exps_sum * np.array([-1.0, 0.0])
            grad[k] += exps[2*n + 2*n + k] / exps_sum * np.array([0.0, 1.0])
            grad[k] += exps[2*n + 3*n + k] / exps_sum * np.array([0.0, -1.0])
        
        # Objective is -r_soft, so gradient of objective = -gradient of r_soft
        # But we want to maximize r_soft, so we go in direction of grad of r_soft
        return -r_soft, -grad  # negative for minimization
    
    lr = lr_init
    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        obj, grad = _objective_and_grad(pts)
        
        # Normalized gradient step
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            pts = pts + lr * grad / grad_norm * 0.01  # scaled step
        
        # Project back into square
        pts[:, 0] = np.clip(pts[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
        pts[:, 1] = np.clip(pts[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
        
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
        
        # Decay learning rate
        lr *= 0.999
        if lr < lr_min:
            lr = lr_min
    
    return pts


def _delaunay_flip_optimize(pts, max_flips=20000):
    """
    Use Delaunay triangulation to identify edges and try to maximize
    the minimum edge length via local moves.
    """
    pts = pts.copy().astype(np.float64)
    best_r = _compute_radius(pts)
    
    for flip_iter in range(max_flips):
        if time.time() - _START_TIME > _TIME_LIMIT - 15:
            break
        
        # Compute current radius
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
        
        # Build Delaunay triangulation
        try:
            tri = Delaunay(pts)
        except Exception:
            break
        
        # Get all edges from triangulation
        simplices = tri.simplices
        edges = set()
        for s in simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = sorted((s[i], s[j]))
                    edges.add((a, b))
        
        # Find the shortest edge
        min_edge_dist = np.inf
        min_edge = None
        for a, b in edges:
            d = np.linalg.norm(pts[a] - pts[b])
            if d < min_edge_dist:
                min_edge_dist = d
                min_edge = (a, b)
        
        if min_edge is None:
            break
        
        # Also check wall distances
        wall_dist = np.min([
            np.min(pts[:, 0]) + 0.5,
            0.5 - np.max(pts[:, 0]),
            np.min(pts[:, 1]) + 0.5,
            0.5 - np.max(pts[:, 1]),
        ])
        
        # If wall is the bottleneck, move the closest-to-wall point inward
        if wall_dist < min_edge_dist / 2:
            # Find point closest to wall
            dists_to_walls = np.min([
                pts[:, 0] + 0.5,
                0.5 - pts[:, 0],
                pts[:, 1] + 0.5,
                0.5 - pts[:, 1],
            ], axis=0)
            idx = np.argmin(dists_to_walls)
            # Move inward
            pt = pts[idx]
            # Determine which wall is closest
            d_left = pt[0] + 0.5
            d_right = 0.5 - pt[0]
            d_bottom = pt[1] + 0.5
            d_top = 0.5 - pt[1]
            min_d = min(d_left, d_right, d_bottom, d_top)
            if min_d == d_left:
                pts[idx, 0] += 0.005
            elif min_d == d_right:
                pts[idx, 0] -= 0.005
            elif min_d == d_bottom:
                pts[idx, 1] += 0.005
            else:
                pts[idx, 1] -= 0.005
            pts[idx] = np.clip(pts[idx], -0.5, 0.5)
        else:
            # Move the two endpoints of the shortest edge apart
            a, b = min_edge
            diff = pts[b] - pts[a]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                pts[b] += np.random.randn(2) * 0.01
            else:
                direction = diff / d
                move = 0.005
                new_a = pts[a] - move * direction
                new_b = pts[b] + move * direction
                # Check bounds
                if -0.5 <= new_a[0] <= 0.5 and -0.5 <= new_a[1] <= 0.5 and \
                   -0.5 <= new_b[0] <= 0.5 and -0.5 <= new_b[1] <= 0.5:
                    pts[a] = new_a
                    pts[b] = new_b
                else:
                    # Try smaller move
                    for frac in [0.5, 0.25, 0.1]:
                        move = 0.005 * frac
                        new_a = pts[a] - move * direction
                        new_b = pts[b] + move * direction
                        if -0.5 <= new_a[0] <= 0.5 and -0.5 <= new_a[1] <= 0.5 and \
                           -0.5 <= new_b[0] <= 0.5 and -0.5 <= new_b[1] <= 0.5:
                            pts[a] = new_a
                            pts[b] = new_b
                            break
        
        # Every 100 iterations, do a small gradient refinement
        if flip_iter % 100 == 0 and flip_iter > 0:
            pts = _gradient_refine(pts, max_iters=200)
    
    return pts


def _random_restart(n_restarts=5):
    """Generate several random initial configurations and return the best."""
    best_pts = None
    best_r = -1
    
    np.random.seed(42)
    
    for restart in range(n_restarts):
        if time.time() - _START_TIME > _TIME_LIMIT - 20:
            break
        
        # Different initializations
        if restart == 0:
            # Hexagonal grid
            pts = _generate_hexagonal_grid(N)
        elif restart == 1:
            # Square grid 14x15
            pts = _generate_grid((14, 15))
        elif restart == 2:
            # Square grid 15x14
            pts = _generate_grid((15, 14))
        elif restart == 3:
            # Random
            pts = np.random.uniform(-0.45, 0.45, (N, 2))
        else:
            # Jittered hexagonal
            pts = _generate_hexagonal_grid(N)
            pts += np.random.randn(N, 2) * 0.02
            pts = np.clip(pts, -0.5, 0.5)
        
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    return best_pts


def construct():
    t0 = time.time()
    
    # Step 1: Generate initial configurations
    best_pts = _random_restart(n_restarts=5)
    best_r = _compute_radius(best_pts)
    
    # Step 2: Delaunay-guided optimization
    if best_r > 0 and time.time() - t0 < _TIME_LIMIT - 30:
        try:
            pts_flip = _delaunay_flip_optimize(best_pts, max_flips=15000)
            r_flip = _compute_radius(pts_flip)
            if r_flip > best_r:
                best_r = r_flip
                best_pts = pts_flip
        except Exception:
            pass
    
    # Step 3: Gradient refinement
    if best_r > 0 and time.time() - t0 < _TIME_LIMIT - 15:
        try:
            pts_grad = _gradient_refine(best_pts, max_iters=3000)
            r_grad = _compute_radius(pts_grad)
            if r_grad > best_r:
                best_r = r_grad
                best_pts = pts_grad
        except Exception:
            pass
    
    # Step 4: Final Delaunay pass
    if best_r > 0 and time.time() - t0 < _TIME_LIMIT - 10:
        try:
            pts_flip2 = _delaunay_flip_optimize(best_pts, max_flips=5000)
            r_flip2 = _compute_radius(pts_flip2)
            if r_flip2 > best_r:
                best_r = r_flip2
                best_pts = pts_flip2
        except Exception:
            pass
    
    # Step 5: Final gradient polish
    if best_r > 0 and time.time() - t0 < _TIME_LIMIT - 5:
        try:
            pts_grad2 = _gradient_refine(best_pts, max_iters=1000)
            r_grad2 = _compute_radius(pts_grad2)
            if r_grad2 > best_r:
                best_r = r_grad2
                best_pts = pts_grad2
        except Exception:
            pass
    
    if best_r <= 0:
        # Fallback: simple grid
        rows, cols = 20, 10
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        return grid[:N]
    
    return best_pts
