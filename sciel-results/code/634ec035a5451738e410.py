"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill

def _score(pts):
    """
    Compute the valid packing radius.
    Returns 0 if invalid.
    """
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Container constraint: min_i (1 - |c_i|)
    dists = np.hypot(pts[:, 0], pts[:, 1])
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
        
    # Pairwise constraint: min_{i<j} |c_i - c_j| / 2
    if N < 2:
        return r_cont
        
    # Calculate pairwise distances
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dists_sq = dx**2 + dy**2
    
    # We need min distance between distinct points
    np.fill_diagonal(dists_sq, np.inf)
    min_dist_sq = np.min(dists_sq)
    r_pair = np.sqrt(min_dist_sq) / 2.0
    
    return min(r_cont, r_pair)

def _init_hex():
    """
    Generate initial points using a hexagonal lattice cut to the unit disk.
    """
    best_points = None
    best_score = -1
    
    for spacing in np.linspace(0.2, 0.35, 10):
        k = int(1.0 / spacing) + 2
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0:
                    rows.append((x, y))
        points = np.array(rows)
        
        if len(points) < N:
            continue
            
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii, kind="stable")[:N]
        pts = points[indices]
        
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
            
    if best_points is None:
        pts = np.random.uniform(-0.8, 0.8, (N, 2))
        return pts
        
    return best_points

def _optimize_repulsive(points, max_steps=3000, time_limit=None):
    """
    Optimize packing using repulsive forces.
    """
    pts = points.copy()
    N = len(pts)
    
    step_size = 0.05
    decay = 0.995
    
    start_time = time.time()
    
    for step in range(max_steps):
        if time_limit and time.time() > time_limit:
            break
            
        forces = np.zeros_like(pts)
        
        dx = pts[:, 0:1] - pts[0:1, 0]
        dy = pts[:, 1:2] - pts[0:1, 1]
        dists_sq = dx**2 + dy**2
        dists = np.sqrt(dists_sq)
        
        dists[dists < 1e-6] = 1e-6
        
        inv_d3 = 1.0 / (dists**3)
        np.fill_diagonal(inv_d3, 0.0)
        
        fx = np.sum(dx * inv_d3, axis=1)
        fy = np.sum(dy * inv_d3, axis=1)
        
        forces[:, 0] += fx
        forces[:, 1] += fy
        
        r = np.hypot(pts[:, 0], pts[:, 1])
        mask = r > 1.0
        if np.any(mask):
            strength = (r[mask] - 1.0)
            norm = r[mask]
            forces[mask, 0] -= pts[mask, 0] / norm * strength
            forces[mask, 1] -= pts[mask, 1] / norm * strength
            
        pts += step_size * forces
        step_size *= decay
        
    return pts

def construct():
    """
    Construct the optimal packing of N circles.
    """
    best_pts = None
    best_score = -1.0
    
    # 1. Start with Hex Lattice
    init_pts = _init_hex()
    optimized_pts = _optimize_repulsive(init_pts, max_steps=5000, time_limit=_DEADLINE)
    score = _score(optimized_pts)
    
    if score > best_score:
        best_score = score
        best_pts = optimized_pts.copy()
        
    # 2. Try Random Restarts if time permits
    while time.time() < _DEADLINE - 5.0:
        # Jitter the initial points
        jittered = _init_hex()
        jittered += np.random.normal(0, 0.05, jittered.shape)
        
        opt_pts = _optimize_repulsive(jittered, max_steps=3000, time_limit=_DEADLINE)
        s = _score(opt_pts)
        
        if s > best_score:
            best_score = s
            best_pts = opt_pts.copy()
            
        if best_score > 0.1258:
            break
            
    if best_pts is None:
        return _init_hex()
        
    # Final projection to ensure validity
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / r[mask] * (1.0 - 1e-9)
        
    return best_pts

# EVOLVE-BLOCK-END
