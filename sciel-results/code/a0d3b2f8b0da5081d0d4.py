import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _hex_lattice_init():
    """Generate a hexagonal lattice initialization with 50 points."""
    spacing = 0.25
    while True:
        points = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if x*x + y*y <= (1 - spacing/2)**2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:N]
            return points[idx]
        spacing *= 0.95

def _ring_init(n_rings=4):
    """Generate a ring-based initialization with 50 points."""
    points = [[0.0, 0.0]]
    # Distribute remaining 49 points across rings
    remaining = N - 1
    pts_per_ring = []
    for k in range(1, n_rings + 1):
        # Rough estimate of how many fit on ring k
        count = int(2 * np.pi * k * 0.35)
        pts_per_ring.append(count)
    
    # Adjust to get exactly N-1 points
    total = sum(pts_per_ring)
    while total < remaining:
        pts_per_ring[-1] += 1
        total += 1
    while total > remaining:
        pts_per_ring[-1] -= 1
        total -= 1
    
    for k, count in enumerate(pts_per_ring):
        if count <= 0:
            continue
        radius = 0.85 * (k + 1) / n_rings
        angles = 2 * np.pi * np.arange(count) / count + np.pi / count  # offset to avoid alignment
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    return np.array(points[:N])

def _compute_score(points):
    """Compute the score: min(min_pairwise_dist/2, min(1 - |c_i|))."""
    n = len(points)
    if n != N:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms > 1.0):
        return 0.0
    
    min_boundary = 1.0 - np.max(norms)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    min_dist = np.sqrt(np.min(dist_sq))
    
    return min(min_dist / 2.0, min_boundary)

def _soft_min_energy(x, beta=50.0):
    """
    Compute a smooth surrogate for -min(d_ij/2, 1-|c_i|) using log-sum-exp.
    Returns a scalar to be minimized.
    """
    pts = x.reshape(N, 2)
    
    # Boundary term: 1 - |c_i|
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms  # shape (N,)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dists = np.sqrt(dist_sq)  # shape (N, N)
    
    # We want to maximize r = min(min_ij d_ij/2, min_i (1-|c_i|))
    # Let a = min_ij d_ij/2, b = min_i (1-|c_i|)
    # We approximate min(a, b) with a soft version.
    
    # Soft min of all pairwise distances / 2
    # Use log-sum-exp trick: -log(sum(exp(-beta * v))) / beta approximates min(v)
    # For pairwise distances:
    pair_vals = dists / 2.0  # (N, N)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    pair_vals_masked = np.where(mask, pair_vals, np.inf)
    
    # Soft min of pair values
    min_pair = -np.log(np.sum(np.exp(-beta * pair_vals_masked))) / beta
    
    # Soft min of boundary values
    min_boundary = -np.log(np.sum(np.exp(-beta * boundary))) / beta
    
    # Soft min of min_pair and min_boundary
    # min(a, b) approx -log(exp(-beta*a) + exp(-beta*b)) / beta
    r_soft = -np.log(np.exp(-beta * min_pair) + np.exp(-beta * min_boundary)) / beta
    
    # We want to maximize r_soft, so minimize -r_soft
    return -r_soft

def _compute_score_fast(points):
    """Fast score computation for optimization."""
    n = len(points)
    if n != N:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms > 1.0):
        return 0.0
    
    min_boundary = 1.0 - np.max(norms)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    min_dist = np.sqrt(np.min(dist_sq))
    
    return min(min_dist / 2.0, min_boundary)

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin from 150s limit
    
    # Start with multiple initializations
    init_points = []
    init_points.append(_hex_lattice_init())
    init_points.append(_ring_init(4))
    init_points.append(_ring_init(5))
    
    best_score = 0.0
    best_points = init_points[0].copy()
    
    for pts in init_points:
        s = _compute_score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    bounds = [(-1, 1)] * (2 * N)
    rng = np.random.default_rng(42)
    
    # Local optimization using smooth soft-min objective
    def run_local_opt(x0, maxiter=150, beta=50.0):
        try:
            def obj(x):
                return _soft_min_energy(x, beta)
            
            res = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            new_points = res.x.reshape(N, 2)
            score = _compute_score_fast(new_points)
            if score > 0:
                return new_points, score
        except Exception:
            pass
        return None, -np.inf
    
    # Stage 1: Optimize from best initial
    if time.time() - start_time < deadline - 10:
        pts, score = run_local_opt(best_points.flatten(), maxiter=200, beta=50.0)
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Stage 2: Random restarts with ring-like structure
    n_restarts = 40
    for i in range(n_restarts):
        if time.time() - start_time > deadline - 10:
            break
        
        # Generate structured random initialization
        if i % 2 == 0:
            # Ring-based with random perturbations
            base = _ring_init(4 + (i % 3))
            noise = rng.normal(0, 0.05, size=base.shape)
            x0 = (base + noise).flatten()
        else:
            # Random uniform in disk
            r = rng.random(N) ** 0.5
            theta = 2 * np.pi * rng.random(N)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            x0 = np.column_stack((x, y)).flatten()
        
        pts, score = run_local_opt(x0, maxiter=100, beta=40.0)
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts.copy()
            
            # Quick polish
            if time.time() - start_time < deadline - 5:
                pts2, score2 = run_local_opt(best_points.flatten(), maxiter=50, beta=60.0)
                if pts2 is not None and score2 > best_score:
                    best_score = score2
                    best_points = pts2.copy()
    
    # Stage 3: Binary search polish - refine the radius
    # The idea: we have a solution with score r. We can try to see if we can
    # slightly improve by doing a more precise local optimization.
    if time.time() - start_time < deadline - 3:
        # Do a final high-precision polish
        pts, score = run_local_opt(best_points.flatten(), maxiter=300, beta=100.0)
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts.copy()
        
        # One more pass with even tighter tolerance
        if time.time() - start_time < deadline - 2:
            def obj_tight(x):
                return _soft_min_energy(x, beta=200.0)
            
            try:
                res = minimize(
                    obj_tight,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
                )
                new_pts = res.x.reshape(N, 2)
                s = _compute_score_fast(new_pts)
                if s > best_score:
                    best_score = s
                    best_points = new_pts.copy()
            except Exception:
                pass
    
    # Final validation
    final_score = _compute_score(best_points)
    if final_score <= 0:
        return _hex_lattice_init()
    
    return best_points
