"""Improved sphere packing for 50 equal spheres in a unit cube."""
import numpy as np
import time

N = 50

def _compute_radius(pts):
    """Compute the maximum common radius for given sphere centers."""
    n = len(pts)
    # Pairwise distances (vectorized)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    min_wall = np.min(wall_dists)
    return min(min_dist / 2.0, min_wall)

def _sa_optimize(pts, time_limit, seed=0):
    """
    Vectorized Simulated Annealing optimization.
    """
    start_time = time.time()
    n = len(pts)
    pts = pts.copy()
    current_r = _compute_radius(pts)
    best_r = current_r
    best_pts = pts.copy()
    
    # SA parameters
    T = 0.05
    T_min = 1e-5
    alpha = 0.999
    step_scale = 0.1
    
    rng = np.random.default_rng(seed)
    
    while time.time() - start_time < time_limit and T > T_min:
        # Generate random move
        # Move a random subset of points
        num_moves = rng.integers(1, max(2, n // 10) + 1)
        indices = rng.choice(n, size=num_moves, replace=False)
        
        # Random displacement
        displacements = rng.normal(0, step_scale, size=(num_moves, 3))
        
        new_pts = pts.copy()
        new_pts[indices] += displacements
        
        # Clip to cube
        new_pts = np.clip(new_pts, -0.4999, 0.4999)
        
        new_r = _compute_radius(new_pts)
        
        # Acceptance probability
        delta_r = new_r - current_r
        if delta_r > 0 or rng.random() < np.exp(delta_r / T):
            pts = new_pts
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
        
        # Cooling
        T *= alpha
        
        # Adapt step size occasionally
        if rng.random() < 0.01:
            if current_r > best_r * 0.95:
                step_scale *= 0.8
            else:
                step_scale *= 1.2
            step_scale = np.clip(step_scale, 0.01, 0.5)
            
    return best_pts, best_r

def _generate_grid_start():
    """Generate a 50-point starting configuration from a jittered 4x4x4 grid."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Jitter to break symmetry
    np.random.seed(42)
    grid += np.random.randn(*grid.shape) * 0.02
    grid = np.clip(grid, -0.49, 0.49)
    return grid[:N].copy()

def _generate_fcc_start():
    """Generate starting configuration from FCC lattice points."""
    # FCC lattice: points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) mod 1
    # Scale to fit in [-0.5, 0.5]^3
    points = []
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                for offset in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]])
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        points.append(p)
    points = np.array(points)
    # Keep only 50
    if len(points) > N:
        # Randomly select 50
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        # Pad with random points
        extra = np.random.randn(N - len(points), 3) * 0.3
        points = np.vstack([points, extra])
    return points[:N].copy()

def _generate_random_repulsive():
    """Generate random points with repulsion to avoid overlaps."""
    np.random.seed(123)
    pts = np.random.randn(N, 3) * 0.4
    # Simple repulsion iterations
    for _ in range(100):
        for i in range(N):
            for j in range(i + 1, N):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 0.15:
                    f = d / max(dist, 1e-10) * 0.01
                    pts[i] += f
                    pts[j] -= f
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def construct():
    start_time = time.time()
    time_limit = 120.0  # Safety margin before 150s kill
    
    candidates = []
    
    # 1. Jittered grid
    candidates.append(_generate_grid_start())
    
    # 2. FCC-like
    candidates.append(_generate_fcc_start())
    
    # 3. Random repulsive
    candidates.append(_generate_random_repulsive())
    
    # Add a few more random starts
    for i in range(3):
        np.random.seed(100 + i)
        pts = np.random.randn(N, 3) * 0.4
        pts = np.clip(pts, -0.49, 0.49)
        candidates.append(pts)
        
    best_pts = None
    best_r = 0
    
    num_candidates = len(candidates)
    
    for i, init_pts in enumerate(candidates):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 10.0  # 10s safety margin for final steps
        
        if remaining < 5.0:
            break
            
        # Distribute remaining time among remaining candidates
        # Simple heuristic: give each candidate a fixed chunk, e.g., 15s, or proportional.
        # Let's try 15s per candidate.
        t_budget = min(15.0, remaining / max(1, num_candidates - i))
        
        if t_budget < 1.0:
            continue
            
        pts, r = _sa_optimize(init_pts, t_budget, seed=1000 + i)
        
        if r > best_r:
            best_r = r
            best_pts = pts
    
    # Final polish on best result
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 2.0
    if remaining > 5.0 and best_pts is not None:
        best_pts, best_r = _sa_optimize(best_pts, remaining, seed=9999)
    
    # Ensure validity
    if best_pts is None:
        best_pts = _generate_grid_start()
    
    best_pts = np.clip(best_pts, -0.4999999, 0.4999999)
    
    return best_pts
