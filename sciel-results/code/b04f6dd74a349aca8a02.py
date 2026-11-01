"""Improved packing of 50 equal spheres in a unit cube."""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time

N = 50
CUBE_HALF = 0.5
DEADLINE_MARGIN = 15.0  # safety margin before 150s kill


def _compute_radius(pts):
    """Compute the common radius for a given set of centers."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check boundaries
    max_coord = np.max(np.abs(pts), axis=1)
    if np.any(max_coord > CUBE_HALF + 1e-12):
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist <= 1e-12:
        return 0.0
    r_pair = min_dist / 2.0
    r_wall = CUBE_HALF - np.max(max_coord)
    return min(r_pair, r_wall)


def _pair_dist_matrix(pts):
    """Compute pairwise distance matrix."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return dists


def _soft_min(vals, beta=50.0):
    """Soft minimum approximation (numerically stable)."""
    vals = np.asarray(vals)
    m = np.min(vals)
    return m - 1.0 / beta * np.log(np.sum(np.exp(-beta * (vals - m))))


def _objective_and_grad(pts, beta=50.0):
    """Vectorized objective and gradient for soft-min radius maximization."""
    pts = np.asarray(pts).reshape(N, 3)
    
    # Pairwise distances (vectorized)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall margins
    wall_margins = CUBE_HALF - np.abs(pts)  # shape (N, 3)
    wall_margins_flat = wall_margins.flatten()
    
    # Upper triangle indices for pairs
    tri_i, tri_j = np.triu_indices(N, k=1)
    pair_radii = dists[tri_i, tri_j] / 2.0
    
    # Combined radii
    all_radii = np.concatenate([pair_radii, wall_margins_flat])
    
    # Soft min (stable)
    m = np.min(all_radii)
    exp_term = np.exp(-beta * (all_radii - m))
    sum_exp = np.sum(exp_term)
    r_soft = m - (1.0 / beta) * np.log(sum_exp)
    obj = -r_soft
    
    # Weights for gradient
    weights = exp_term / sum_exp
    
    # Vectorized gradient computation
    grad = np.zeros_like(pts)
    
    # Pair contributions: for pair (i,j), weight w, contribution w*(p_i-p_j)/(2*d_ij) to grad[i]
    # Use np.add.at for accumulation
    pair_dists = dists[tri_i, tri_j]
    pair_weights = weights[:len(tri_i)]
    
    # Direction vectors
    dir_vecs = (pts[tri_i] - pts[tri_j]) / (2.0 * pair_dists[:, np.newaxis])
    
    # Accumulate into grad
    np.add.at(grad, tri_i, pair_weights[:, np.newaxis] * dir_vecs)
    np.add.at(grad, tri_j, -pair_weights[:, np.newaxis] * dir_vecs)
    
    # Wall contributions: for wall term (i, a), weight w, contribution -w*sign(p_i[a]) to grad[i,a]
    wall_weights = weights[len(tri_i):]
    signs = np.sign(pts)
    # Reshape wall weights to (N, 3)
    wall_w = wall_weights.reshape(N, 3)
    grad -= wall_w * signs
    
    # Objective gradient is -grad (since we minimize -r_soft)
    obj_grad = -grad
    
    return obj, obj_grad.ravel()


def _lloyd_step(pts, dists):
    """One Lloyd iteration: move each point to centroid of its Voronoi cell."""
    try:
        delaunay = Delaunay(pts)
        # Compute Voronoi vertices
        voronoi = delaunay.coplanar
        # Use scipy's Voronoi for better support
        from scipy.spatial import Voronoi
        vor = Voronoi(pts)
        
        new_pts = pts.copy()
        for i in range(N):
            # Get vertices of Voronoi cell
            if vor.regions[i] is None or len(vor.regions[i]) == 0:
                continue
            verts_idx = vor.regions[i]
            if -1 in verts_idx:
                continue
            verts = vor.vertices[verts_idx]
            
            # Compute centroid
            centroid = np.mean(verts, axis=0)
            
            # Clip to cube
            centroid = np.clip(centroid, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
            
            # Check if centroid is valid (inside cube and not too close to others)
            new_pts[i] = centroid
        
        # Clip all points
        new_pts = np.clip(new_pts, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        return new_pts
    except Exception:
        return pts


def _lloyd_optimize(pts, max_iters=50, time_budget=10.0):
    """Run Lloyd iterations with time budget."""
    start = time.time()
    for _ in range(max_iters):
        if time.time() - start > time_budget:
            break
        dists = _pair_dist_matrix(pts)
        pts = _lloyd_step(pts, dists)
    return pts


def _repulsion_anneal(pts, start_temp=0.1, end_temp=0.001, n_steps=100, time_budget=15.0):
    """Repulsion annealing to escape local minima."""
    start = time.time()
    rng = np.random.default_rng(123)
    for step in range(n_steps):
        if time.time() - start > time_budget:
            break
        t = step / max(n_steps - 1, 1)
        temp = start_temp * (end_temp / start_temp) ** t
        
        # Compute repulsive forces
        dists = _pair_dist_matrix(pts)
        forces = np.zeros_like(pts)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-12:
                    continue
                # Repulsive force: k/d^2 in direction away from j
                force_mag = temp / (d ** 2)
                forces[i] += (pts[i] - pts[j]) / d * force_mag
        
        # Add wall repulsion
        for i in range(N):
            for a in range(3):
                margin = CUBE_HALF - np.abs(pts[i, a])
                if margin < 0.1:
                    forces[i, a] += temp * (1.0 - margin / 0.1) * np.sign(pts[i, a])
        
        # Metropolis acceptance
        new_pts = pts + forces * 0.01
        new_pts = np.clip(new_pts, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        
        r_old = _compute_radius(pts)
        r_new = _compute_radius(new_pts)
        
        if r_new > r_old or rng.random() < np.exp((r_new - r_old) / max(temp, 1e-10)):
            pts = new_pts
    
    return pts


def _smooth_polish(pts, beta_start=50.0, beta_end=500.0, n_iters=30, time_budget=30.0):
    """Smooth optimization with annealed beta."""
    start = time.time()
    bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
    
    for it in range(n_iters):
        if time.time() - start > time_budget:
            break
        t = it / max(n_iters - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        def obj_func(p):
            f, g = _objective_and_grad(p, beta)
            return f, g
        
        try:
            res = minimize(obj_func, pts.ravel(), jac=True, bounds=bounds,
                          method='L-BFGS-B', options={'maxiter': 20, 'gtol': 1e-7})
            pts_new = res.x.reshape(N, 3)
            pts_new = np.clip(pts_new, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
            
            r_new = _compute_radius(pts_new)
            r_old = _compute_radius(pts)
            if r_new >= r_old:
                pts = pts_new
        except Exception:
            pass
    
    return pts


def _hard_local_search(pts, time_budget=5.0):
    """Hard local search using exact distance calculations."""
    start = time.time()
    while time.time() - start < time_budget:
        improved = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 0.2:  # Only consider close pairs
                    # Try to move i and j apart
                    direction = (pts[i] - pts[j]) / d if d > 1e-12 else np.array([1.0, 0.0, 0.0])
                    for delta in [0.001, 0.002, 0.005]:
                        new_pts = pts.copy()
                        new_pts[i] += direction * delta
                        new_pts[j] -= direction * delta
                        new_pts = np.clip(new_pts, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
                        
                        r_old = _compute_radius(pts)
                        r_new = _compute_radius(new_pts)
                        if r_new > r_old:
                            pts = new_pts
                            improved = True
                            break
        if not improved:
            break
    return pts


def _seed_grid():
    """Grid-based initial placement."""
    rng = np.random.default_rng(42)
    m = 4  # 4x4x4 = 64 > 50
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:N].copy()
    pts += rng.normal(0, 0.01, size=pts.shape)
    return np.clip(pts, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)


def _seed_fcc():
    """FCC-like initial placement."""
    rng = np.random.default_rng(99)
    # FCC lattice points in a cube
    a = 1.0 / 4.0  # lattice constant
    pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x, y, z = i * a, j * a, k * a
                pts.append([x, y, z])
                pts.append([x + a/2, y + a/2, z])
                pts.append([x + a/2, y, z + a/2])
                pts.append([x, y + a/2, z + a/2])
    pts = np.array(pts)
    # Keep first N points
    pts = pts[:N].copy()
    pts += rng.normal(0, 0.005, size=pts.shape)
    return np.clip(pts, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)


def _seed_bcc():
    """BCC-like initial placement."""
    rng = np.random.default_rng(123)
    a = 1.0 / 3.0  # lattice constant
    pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x, y, z = i * a, j * a, k * a
                pts.append([x, y, z])
                pts.append([x + a/2, y + a/2, z + a/2])
    pts = np.array(pts)
    # Keep first N points
    pts = pts[:N].copy()
    pts += rng.normal(0, 0.005, size=pts.shape)
    return np.clip(pts, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)


def _seed_shell_core():
    """Shell-core initial placement: spheres near corners and face centers."""
    rng = np.random.default_rng(456)
    pts = []
    # Corners
    for x in [-CUBE_HALF + 0.1, CUBE_HALF - 0.1]:
        for y in [-CUBE_HALF + 0.1, CUBE_HALF - 0.1]:
            for z in [-CUBE_HALF + 0.1, CUBE_HALF - 0.1]:
                pts.append([x, y, z])
    # Face centers
    for axis in range(3):
        for sign in [-1, 1]:
            for i in range(2):
                for j in range(2):
                    pos = [0.0, 0.0, 0.0]
                    pos[axis] = sign * (CUBE_HALF - 0.1)
                    if axis == 0:
                        pos[1] = -0.25 + i * 0.5
                        pos[2] = -0.25 + j * 0.5
                    elif axis == 1:
                        pos[0] = -0.25 + i * 0.5
                        pos[2] = -0.25 + j * 0.5
                    else:
                        pos[0] = -0.25 + i * 0.5
                        pos[1] = -0.25 + j * 0.5
                    pts.append(pos)
    # Core
    while len(pts) < N:
        pts.append(rng.uniform(-CUBE_HALF + 0.15, CUBE_HALF - 0.15, size=3).tolist())
    pts = np.array(pts[:N])
    pts += rng.normal(0, 0.005, size=pts.shape)
    return np.clip(pts, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)


def _seed_random():
    """Random initial placement."""
    rng = np.random.default_rng(77)
    pts = rng.uniform(-CUBE_HALF + 0.05, CUBE_HALF - 0.05, size=(N, 3))
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN  # 135s effective
    
    # Try multiple seeds and keep the best
    seeds = [_seed_grid(), _seed_fcc(), _seed_bcc(), _seed_shell_core(), _seed_random()]
    
    best_pts = None
    best_r = 0.0
    
    for seed_pts in seeds:
        if time.time() > deadline - 20:
            break
        
        pts = seed_pts.copy()
        
        # Phase 1: Lloyd equalization
        if time.time() < deadline - 25:
            pts = _lloyd_optimize(pts, max_iters=30, time_budget=8.0)
        
        # Phase 2: Repulsion annealing
        if time.time() < deadline - 20:
            pts = _repulsion_anneal(pts, start_temp=0.05, end_temp=0.001, 
                                    n_steps=80, time_budget=12.0)
        
        # Phase 3: Smooth polish
        if time.time() < deadline - 10:
            pts = _smooth_polish(pts, beta_start=30.0, beta_end=400.0,
                                n_iters=25, time_budget=20.0)
        
        # Phase 4: Final high-precision polish
        if time.time() < deadline - 5:
            pts = _smooth_polish(pts, beta_start=400.0, beta_end=1000.0,
                                n_iters=10, time_budget=8.0)
        
        # Phase 5: Hard local search
        if time.time() < deadline - 2:
            pts = _hard_local_search(pts, time_budget=3.0)
        
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _seed_grid()
    
    return best_pts
