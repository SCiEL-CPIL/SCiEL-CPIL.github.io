import time
import numpy as np
from scipy.spatial import cKDTree

N = 200
HALF = 0.5
SEED = 1234567

def _radius_of(c):
    """Compute packing radius for centers c (N,2)."""
    w = HALF - np.abs(c)
    wall = np.min(w)
    d = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    pair = np.sqrt(np.min(dist_sq)) / 2.0
    return float(min(wall, pair))

def _hexagonal_seed(n, r_target=0.036):
    """Generate a hexagonal packing seed with approximately n points."""
    # Hexagonal packing: rows of points with vertical spacing sqrt(3)*r and horizontal 2r
    # Offset alternate rows by r
    r = r_target
    h_spacing = 2 * r
    v_spacing = r * np.sqrt(3)
    
    # Generate a grid of points in hexagonal arrangement
    # Start from bottom-left, fill up
    pts = []
    y = -HALF + r
    row = 0
    while y <= HALF - r and len(pts) < n + 50:
        x_offset = 0.0 if row % 2 == 0 else r
        x = -HALF + r + x_offset
        while x <= HALF - r and len(pts) < n + 50:
            pts.append([x, y])
            x += h_spacing
        y += v_spacing
        row += 1
    
    pts = np.array(pts)
    if len(pts) < n:
        # Pad with random points if needed
        rng = np.random.default_rng(42)
        extra = rng.uniform(-HALF + r, HALF - r, size=(n - len(pts), 2))
        pts = np.vstack([pts, extra])
    
    # Shuffle and take first n to avoid bias
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(pts))[:n]
    return pts[idx]

def _repulsion_force(c, k=50.0):
    """Compute repulsion forces between circles."""
    n = len(c)
    forces = np.zeros_like(c)
    
    # Pairwise distances
    d = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    np.fill_diagonal(dist_sq, 1e-10)
    dist = np.sqrt(dist_sq)
    
    # Repulsion: F = k / dist^2 * direction
    # Avoid division by zero
    mask = dist > 1e-8
    inv_dist_sq = np.where(mask, 1.0 / np.where(dist_sq < 1e-10, 1e-10, dist_sq), 0.0)
    force_mag = k * inv_dist_sq
    force_mag[~mask] = 0.0
    
    # Force direction: d / dist
    dir_vec = d / np.where(dist > 1e-8, dist, 1.0)[:, :, None]
    dir_vec[~mask] = 0.0
    
    # Net force on each particle: sum of forces from all others
    # Force on i from j is in direction (c_i - c_j) if we want repulsion
    # d[i,j] = c_i - c_j, so force on i from j is force_mag[i,j] * dir_vec[i,j]
    # But we want to push i away from j, so force on i is in direction (c_i - c_j)
    # which is exactly dir_vec[i,j] when d = c_i - c_j
    
    forces = np.sum(force_mag[:, :, None] * dir_vec, axis=1)
    
    # Wall forces: push back if too close to walls
    wall_margin = 0.01
    for i in range(n):
        for k_coord in range(2):
            if c[i, k_coord] < -HALF + wall_margin:
                forces[i, k_coord] += k * 0.1
            elif c[i, k_coord] > HALF - wall_margin:
                forces[i, k_coord] -= k * 0.1
    
    return forces

def _local_optimize(c, max_iter=5000, step_scale=0.01, deadline=None, t_start=None):
    """Local optimization using gradient ascent on radius with repulsion forces."""
    if deadline is None:
        deadline = 150.0
    if t_start is None:
        t_start = time.time()
    
    c = c.copy()
    best_c = c.copy()
    best_r = _radius_of(c)
    
    # Use a simple force-directed approach
    # We want to maximize the minimum distance, so we push apart close circles
    
    for it in range(max_iter):
        if time.time() - t_start > deadline - 5.0:
            break
        
        # Compute forces
        forces = _repulsion_force(c, k=1.0)
        
        # Adaptive step size
        step = step_scale / (1.0 + 0.01 * it)
        
        # Update positions
        c_new = c + step * forces
        c_new = np.clip(c_new, -HALF + 1e-9, HALF - 1e-9)
        
        # Check if this improves the radius
        r_new = _radius_of(c_new)
        
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            c = c_new
        else:
            # Try smaller step
            c_new = c + 0.5 * step * forces
            c_new = np.clip(c_new, -HALF + 1e-9, HALF - 1e-9)
            r_new = _radius_of(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new
    
    return best_c, best_r

def _anneal_optimize(c, t_start, deadline):
    """Simulated annealing style optimization."""
    c = c.copy()
    best_c = c.copy()
    best_r = _radius_of(c)
    
    T = 0.05  # Initial temperature
    T_min = 0.001
    cooling = 0.995
    
    while time.time() - t_start < deadline - 5.0 and T > T_min:
        # Generate perturbation
        scale = T
        c_perturbed = c + np.random.default_rng().normal(0, scale, c.shape)
        c_perturbed = np.clip(c_perturbed, -HALF + 1e-9, HALF - 1e-9)
        
        r_perturbed = _radius_of(c_perturbed)
        r_current = _radius_of(c)
        
        # Accept if better, or with probability exp((r_new - r_old) / T)
        if r_perturbed > r_current:
            c = c_perturbed
            if r_perturbed > best_r:
                best_r = r_perturbed
                best_c = c.copy()
        else:
            delta = r_perturbed - r_current
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    c = c_perturbed
        
        T *= cooling
    
    return best_c, best_r

def construct():
    t_start = time.time()
    deadline = 145.0  # Safety margin
    
    # Start with hexagonal seed
    c = _hexagonal_seed(N, r_target=0.036)
    best_c = c.copy()
    best_r = _radius_of(best_c)
    
    # Local optimization
    c_opt, r_opt = _local_optimize(c, max_iter=2000, step_scale=0.005, deadline=deadline, t_start=t_start)
    if r_opt > best_r:
        best_r = r_opt
        best_c = c_opt
    
    # Try multiple restarts with different seeds
    rng = np.random.default_rng(SEED)
    for restart in range(10):
        if time.time() - t_start > deadline - 10.0:
            break
        
        # Generate a new hexagonal seed with slight perturbation
        c_seed = _hexagonal_seed(N, r_target=0.036)
        c_seed += rng.normal(0, 0.005, c_seed.shape)
        c_seed = np.clip(c_seed, -HALF + 1e-6, HALF - 1e-6)
        
        c_opt, r_opt = _local_optimize(c_seed, max_iter=1000, step_scale=0.005, deadline=deadline, t_start=t_start)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt
    
    # Final polish with annealing if time permits
    if time.time() - t_start < deadline - 15.0:
        c_anneal, r_anneal = _anneal_optimize(best_c, t_start, deadline)
        if r_anneal > best_r:
            best_r = r_anneal
            best_c = c_anneal
    
    # Ensure valid output
    best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
    if not np.all(np.isfinite(best_c)):
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
    
    return best_c

if __name__ == "__main__":
    c = construct()
    print("shape", c.shape, "radius", _radius_of(c))
