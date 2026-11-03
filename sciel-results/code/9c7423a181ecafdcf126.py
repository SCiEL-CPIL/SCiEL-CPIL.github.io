"""Pack 200 equal circles in a unit square to maximize common radius.

Uses a combination of:
- Force-directed layout for initial placement
- Soft-min gradient optimization for smooth refinement
- Simulated annealing on exact objective to escape local minima
- Final high-precision polish
"""
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

N = 200
_START_TIME = time.time()
_TIME_LIMIT = 140  # seconds, leave margin before 150s kill


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


def _compute_radius_fast(pts):
    """Faster radius computation using numpy operations."""
    pts = np.asarray(pts, dtype=np.float64)
    if np.any(~np.isfinite(pts)):
        return 0.0
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return 0.0
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    # Use cdist for pairwise distances
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    r = min(wall, min_pair / 2.0)
    return max(r, 0.0)


def _generate_hexagonal_grid(n):
    """Generate a hexagonal (close-packing) grid arrangement."""
    # For n=200, try to find good row/col configuration
    # Hexagonal packing: rows alternate with offset
    # Try different aspect ratios
    best_pts = None
    best_r = -1
    
    for num_rows in range(10, 25):
        num_cols = int(np.ceil(n / num_rows))
        if num_rows * num_cols < n:
            continue
        
        # Spacing for hexagonal packing
        # In hexagonal packing, horizontal spacing = d, vertical = d*sqrt(3)/2
        # We want to fit in [-0.5, 0.5]^2
        # Total width = (num_cols - 1) * d + d = num_cols * d (with offset)
        # Actually: first row has num_cols points, second has num_cols-1 or num_cols+1
        # Let's use: spacing_x such that num_cols * spacing_x <= 1.0
        # spacing_y = spacing_x * sqrt(3) / 2
        # num_rows * spacing_y <= 1.0
        
        # Solve for spacing that fits
        spacing_x_from_width = 1.0 / num_cols
        spacing_y_from_height = 1.0 / (num_rows * np.sqrt(3) / 2)
        spacing_x = min(spacing_x_from_width, spacing_y_from_height)
        
        # Generate points
        pts_list = []
        for row in range(num_rows):
            y = -0.5 + (row + 0.5) * spacing_x * np.sqrt(3) / 2
            offset = 0 if row % 2 == 0 else spacing_x / 2
            for col in range(num_cols):
                x = -0.5 + (col + 0.5) * spacing_x + offset
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pts_list.append([x, y])
        
        if len(pts_list) >= n:
            pts = np.array(pts_list[:n], dtype=np.float64)
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts
    
    if best_pts is None:
        # Fallback: simple grid
        rows = int(np.sqrt(n))
        cols = int(np.ceil(n / rows))
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:n]
    
    return best_pts


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


def _gradient_refine(pts, max_iters=5000, lr_init=0.01, lr_min=0.0001, beta_init=10.0):
    """
    Gradient-based optimization using soft-min approximation.
    Uses log-sum-exp for numerical stability.
    """
    pts = pts.copy().astype(np.float64)
    best_r = _compute_radius(pts)
    
    def _soft_min_and_grad(constraints, beta):
        """Compute soft-min and its gradient using log-sum-exp."""
        # soft_min(x) = -soft_max(-x) = -(-logsumexp(-x/beta)) = logsumexp(-x/beta)
        # where logsumexp(a) = max(a) + log(sum(exp(a - max(a))))
        
        neg_x_beta = -constraints / beta
        max_val = np.max(neg_x_beta)
        if max_val > 500:  # avoid overflow
            neg_x_beta = neg_x_beta - max_val
            max_val = 0
        
        exps = np.exp(neg_x_beta)
        sum_exps = np.sum(exps)
        if sum_exps == 0:
            return -np.min(constraints), np.zeros_like(constraints)
        
        logsumexp = max_val + np.log(sum_exps)
        soft_min_val = -logsumexp * beta
        
        # Gradient: d(soft_min)/dx_i = -exp(-x_i/beta) / sum(exp(-x_j/beta))
        # = -exps[i] / sum_exps
        grad = -exps / sum_exps
        
        return soft_min_val, grad
    
    def _objective_and_grad(pts):
        """Compute negative soft-min radius and its gradient."""
        n = len(pts)
        
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(dist_sq, 1e10)
        dist = np.sqrt(dist_sq)
        
        # Pairwise distances (upper triangle)
        iu = np.triu_indices(n, 1)
        pair_dists = dist[iu] / 2.0
        
        # Wall distances
        wall_dists = np.concatenate([
            pts[:, 0] + 0.5,
            0.5 - pts[:, 0],
            pts[:, 1] + 0.5,
            0.5 - pts[:, 1],
        ])
        
        all_constraints = np.concatenate([pair_dists, wall_dists])
        
        r_soft, constraint_grads = _soft_min_and_grad(all_constraints, beta)
        
        # Build gradient w.r.t. pts
        grad_pts = np.zeros_like(pts)
        
        # Pairwise constraints
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            w = constraint_grads[idx]
            diff_ij = pts[i] - pts[j]
            d = dist[i, j]
            if d > 1e-12:
                # d(||pi-pj||/2)/dpi = (pi-pj)/(2*||pi-pj||)
                grad_pts[i] += w * diff_ij / (2 * d)
                grad_pts[j] += w * (-diff_ij / (2 * d))
        
        # Wall constraints
        # First n: pts[:,0]+0.5, grad=[1,0]
        # Next n: 0.5-pts[:,0], grad=[-1,0]
        # Next n: pts[:,1]+0.5, grad=[0,1]
        # Last n: 0.5-pts[:,1], grad=[0,-1]
        offset = len(iu[0])
        for k in range(n):
            grad_pts[k, 0] += constraint_grads[offset + k]  # left wall
            grad_pts[k, 0] += constraint_grads[offset + n + k] * (-1)  # right wall
            grad_pts[k, 1] += constraint_grads[offset + 2*n + k]  # bottom wall
            grad_pts[k, 1] += constraint_grads[offset + 3*n + k] * (-1)  # top wall
        
        return -r_soft, -grad_pts  # minimize -r_soft
    
    lr = lr_init
    beta = beta_init
    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 15:
            break
        
        obj, grad = _objective_and_grad(pts)
        
        # Normalized gradient step
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-14:
            step_size = lr / grad_norm
            pts = pts + step_size * grad
        
        # Project back into square
        pts[:, 0] = np.clip(pts[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
        pts[:, 1] = np.clip(pts[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
        
        r = _compute_radius_fast(pts)
        if r > best_r:
            best_r = r
        
        # Decay learning rate and increase beta for sharper min
        lr *= 0.998
        beta *= 1.002
        if lr < lr_min:
            lr = lr_min
    
    return pts


def _force_directed_layout(pts, max_iters=5000, repulsion_strength=1.0, attraction_strength=0.1):
    """
    Force-directed layout: repel overlapping pairs, attract to center.
    This is more efficient than Delaunay flip for finding good configurations.
    """
    pts = pts.copy().astype(np.float64)
    best_r = _compute_radius(pts)
    
    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 20:
            break
        
        forces = np.zeros_like(pts)
        
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(dist_sq, 1e10)
        dist = np.sqrt(dist_sq)
        
        # Repulsive force: for each pair, push apart if too close
        # Force magnitude ~ 1/d^2 for close pairs
        for i in range(N):
            for j in range(i+1, N):
                d = dist[i, j]
                if d < 0.1:  # only consider close pairs
                    f = repulsion_strength / (d * d + 1e-10)
                    direction = (pts[i] - pts[j]) / (d + 1e-10)
                    forces[i] += f * direction
                    forces[j] -= f * direction
        
        # Attractive force toward center (weak)
        for i in range(N):
            forces[i] -= attraction_strength * pts[i]
        
        # Apply forces with step size
        step = 0.01
        pts += step * forces
        pts[:, 0] = np.clip(pts[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
        pts[:, 1] = np.clip(pts[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
        
        r = _compute_radius_fast(pts)
        if r > best_r:
            best_r = r
        
        # Decay forces
        repulsion_strength *= 0.999
        attraction_strength *= 0.999
    
    return pts


def _simulated_annealing(pts, max_iters=10000, t_init=0.01, t_min=0.0001, cooling=0.999):
    """
    Simulated annealing on the exact non-smooth objective.
    This can escape local minima that trap smooth methods.
    """
    pts = pts.copy().astype(np.float64)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    current_r = best_r
    
    t = t_init
    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 25:
            break
        
        # Generate a perturbation: move a random point
        idx = np.random.randint(N)
        perturbation = np.random.randn(2) * 0.02
        new_pts = pts.copy()
        new_pts[idx] += perturbation
        new_pts[idx] = np.clip(new_pts[idx], -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_r = _compute_radius_fast(new_pts)
        
        # Accept if better or with probability exp((new_r - current_r)/t)
        if new_r > current_r:
            pts = new_pts
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
        else:
            delta = new_r - current_r
            if delta > -0.001:  # only consider small worsenings
                accept_prob = np.exp(delta / (t + 1e-10))
                if np.random.random() < accept_prob:
                    pts = new_pts
                    current_r = new_r
        
        # Cool down
        t *= cooling
        if t < t_min:
            t = t_min
    
    return best_pts


def _random_restart(n_restarts=5):
    """Generate several random initial configurations and return the best."""
    best_pts = None
    best_r = -1
    
    np.random.seed(42)
    
    for restart in range(n_restarts):
        if time.time() - _START_TIME > _TIME_LIMIT - 30:
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
    
    # Step 2: Force-directed layout for initial optimization
    if best_r > 0 and time.time() - t0 < _TIME_LIMIT - 40:
        try:
            pts_fd = _force_directed_layout(best_pts, max_iters=3000)
            r_fd = _compute_radius(pts_fd)
            if r_fd > best_r:
                best_r = r_fd
                best_pts = pts_fd
        except Exception:
            pass
    
    # Step 3: Gradient refinement with soft-min
    if best_r > 0 and time.time() - t0 < _TIME_LIMIT - 25:
        try:
            pts_grad = _gradient_refine(best_pts, max_iters=5000)
            r_grad = _compute_radius(pts_grad)
            if r_grad > best_r:
                best_r = r_grad
                best_pts = pts_grad
        except Exception:
            pass
    
    # Step 4: Simulated annealing to escape local minima
    if best_r > 0 and time.time() - t0 < _TIME_LIMIT - 15:
        try:
            pts_sa = _simulated_annealing(best_pts, max_iters=8000)
            r_sa = _compute_radius(pts_sa)
            if r_sa > best_r:
                best_r = r_sa
                best_pts = pts_sa
        except Exception:
            pass
    
    # Step 5: Final gradient polish with higher precision
    if best_r > 0 and time.time() - t0 < _TIME_LIMIT - 5:
        try:
            pts_grad2 = _gradient_refine(best_pts, max_iters=3000, beta_init=20.0)
            r_grad2 = _compute_radius(pts_grad2)
            if r_grad2 > best_r:
                best_r = r_grad2
                best_pts = pts_grad2
        except Exception:
            pass
    
    # Final check and return
    if best_r <= 0:
        # Fallback: simple grid
        rows, cols = 20, 10
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        return grid[:N]
    
    # Ensure valid output
    best_pts = np.asarray(best_pts, dtype=np.float64)
    if len(best_pts) != N or best_pts.shape != (N, 2):
        # Fallback
        rows, cols = 20, 10
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        return grid[:N]
    
    if np.any(~np.isfinite(best_pts)):
        rows, cols = 20, 10
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        return grid[:N]
    
    return best_pts
