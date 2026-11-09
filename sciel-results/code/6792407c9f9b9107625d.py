"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import scipy.optimize as opt
import time

N = 50

# EVOLVE-BLOCK-START

def _generate_lattice_points(radius_limit=1.0, spacing=0.1):
    """Generate points on a hexagonal lattice inside a circle."""
    points = []
    max_idx = int(radius_limit / spacing) + 2
    for j in range(-max_idx, max_idx + 1):
        for i in range(-max_idx, max_idx + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3) / 2
            if np.hypot(x, y) <= radius_limit:
                points.append([x, y])
    return np.array(points)

def _score(points):
    """Calculate the packing radius for a given set of points."""
    if len(points) < 2:
        return 0.0
    min_dist = np.inf
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                min_dist = d
    min_rad = np.inf
    for p in points:
        d = 1.0 - np.linalg.norm(p)
        if d < min_rad:
            min_rad = d
    return min(min_dist / 2.0, min_rad)

def _construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    lattice_points = _generate_lattice_points(1.0, 0.1)
    rng = np.random.default_rng(42)
    
    best_r = 0.0
    best_points = None
    
    # Pre-compute indices for pairwise constraints
    pair_indices = []
    for i in range(N):
        for j in range(i + 1, N):
            pair_indices.append((i, j))
    pair_indices = np.array(pair_indices)
    
    def objective(vars):
        return -vars[-1]
    
    def pairwise_constraints(vars):
        pts = vars[:2*N].reshape(N, 2)
        r = vars[-1]
        # Compute all pairwise squared distances
        # Use broadcasting
        # dists_sq[i, j] = |pts[i] - pts[j]|^2
        # We only need upper triangle
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=2)
        # Extract upper triangle
        iu = np.triu_indices(N, k=1)
        dists_sq_upper = dists_sq[iu]
        return dists_sq_upper - 4*r*r
        
    def radial_constraints(vars):
        pts = vars[:2*N].reshape(N, 2)
        r = vars[-1]
        dists_sq = np.sum(pts**2, axis=1)
        return (1 - r)**2 - dists_sq
        
    constraints = [
        {'type': 'ineq', 'fun': pairwise_constraints},
        {'type': 'ineq', 'fun': radial_constraints}
    ]
    
    bounds = [(-1, 1)] * (2 * N) + [(0, 1)]
    
    num_restarts = 8
    attempts = 0
    
    while attempts < num_restarts and time.time() < deadline:
        attempts += 1
        
        if len(lattice_points) >= N:
            idx = rng.choice(len(lattice_points), N, replace=False)
            init_points = lattice_points[idx].copy()
        else:
            init_points = lattice_points.copy()
            while len(init_points) < N:
                p = rng.uniform(-0.9, 0.9, size=2)
                if np.linalg.norm(p) < 0.9:
                    init_points = np.vstack([init_points, p])
        
        init_points += rng.normal(0, 0.02, size=init_points.shape)
        
        r_init = _score(init_points) * 0.95
        if r_init <= 0:
            r_init = 0.05
            
        x0 = np.concatenate([init_points.flatten(), [r_init]])
        
        try:
            res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=constraints, options={'maxiter': 50, 'ftol': 1e-6})
            
            if res.success or -res.fun > best_r:
                r_found = -res.fun
                pts_found = res.x[:2*N].reshape(N, 2)
                
                r_valid = _score(pts_found)
                if r_valid > best_r:
                    best_r = r_valid
                    best_points = pts_found
        except Exception:
            pass

    if best_points is None:
        # Fallback
        lattice_coarse = _generate_lattice_points(1.0, 0.2)
        if len(lattice_coarse) >= N:
            best_points = lattice_coarse[:N]
        else:
            best_points = np.random.uniform(-0.8, 0.8, size=(N, 2))
        best_r = _score(best_points)
        
    return best_points

# EVOLVE-BLOCK-END

def construct():
    return _construct()
