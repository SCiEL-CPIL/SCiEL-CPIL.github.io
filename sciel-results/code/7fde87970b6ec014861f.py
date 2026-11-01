"""Improved packer: hex-lattice init + force-directed repulsion + boundary + Nelder-Mead polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 150.0
SAFETY_MARGIN = 10.0


def _compute_score(pts):
    """Compute the minimum radius for a given set of centers."""
    if pts.shape[0] != N:
        return 0.0
    # Boundary constraint: 1 - |c_i|
    dists_to_origin = np.linalg.norm(pts, axis=1)
    boundary_min = np.min(1.0 - dists_to_origin)
    if boundary_min < 0:
        return 0.0
    # Pairwise distances
    from scipy.spatial.distance import pdist
    pair_dists = pdist(pts, 'euclidean')
    if len(pair_dists) == 0:
        pair_min = 1.0
    else:
        pair_min = np.min(pair_dists)
    return min(boundary_min, pair_min / 2.0)


def _hex_lattice_init(n_points):
    """Generate a hexagonal lattice initial configuration."""
    # Try different spacings to get enough points
    best_pts = None
    for spacing in np.linspace(0.15, 0.35, 20):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n_points:
            # Sort by distance from center, take closest N
            idx = np.argsort(radius[mask])[:n_points]
            pts = points[idx]
            if best_pts is None or _compute_score(pts) > _compute_score(best_pts):
                best_pts = pts
    if best_pts is None:
        # Fallback: uniform random
        best_pts = np.random.uniform(-0.9, 0.9, size=(n_points, 2))
    return best_pts


def _force_optimize(pts, time_budget=60.0):
    """Force-directed optimization to improve packing."""
    start_time = time.time()
    n = len(pts)
    
    # Precompute initial distances
    def compute_forces(positions):
        forces = np.zeros_like(positions)
        for i in range(n):
            for j in range(i + 1, n):
                d = positions[i] - positions[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    # Add small random perturbation
                    d = np.random.uniform(-0.01, 0.01, size=2)
                    dist = np.linalg.norm(d)
                # Repulsive force (inverse square law, stronger when close)
                force_mag = 1.0 / (dist ** 2 + 1e-6)
                force = force_mag * d / dist
                forces[i] += force
                forces[j] -= force
            # Boundary attraction: push back if too close to edge
            r = np.linalg.norm(positions[i])
            if r > 0.95:
                # Pull toward center
                forces[i] -= (positions[i] / r) * (r - 0.95) * 5.0
        return forces
    
    # Simulated annealing style
    T = 0.01
    cooling = 0.995
    step_size = 0.02
    
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    
    while time.time() - start_time < time_budget:
        forces = compute_forces(pts)
        
        # Add noise
        noise = np.random.normal(0, T, size=(n, 2))
        
        # Update positions
        new_pts = pts + step_size * (forces + noise)
        
        # Keep inside circle
        r = np.linalg.norm(new_pts, axis=1)
        mask = r > 0.99
        if np.any(mask):
            new_pts[mask] *= 0.99 / r[mask]
        
        new_score = _compute_score(new_pts)
        
        if new_score > best_score:
            best_pts = new_pts.copy()
            best_score = new_score
        
        # Accept if better or with some probability
        if new_score >= _compute_score(pts) or np.random.random() < np.exp((new_score - _compute_score(pts)) / T):
            pts = new_pts
        
        T *= cooling
        step_size *= 0.999
    
    return best_pts, best_score


def _nelder_mead_polish(pts, time_budget=30.0):
    """Nelder-Mead optimization to polish the solution."""
    start_time = time.time()
    
    def objective(x):
        pts = x.reshape(-1, 2)
        return -_compute_score(pts)
    
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    
    while time.time() - start_time < time_budget:
        res = minimize(objective, pts.flatten(), method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12})
        
        if res.success or res.fun < objective(pts.flatten()):
            new_pts = res.x.reshape(-1, 2)
            new_score = _compute_score(new_pts)
            if new_score > best_score:
                best_pts = new_pts.copy()
                best_score = new_score
            pts = new_pts
        
        # Check if we're stuck
        if _compute_score(pts) >= best_score * (1 - 1e-10):
            break
        
        # Perturb if stuck
        pts += np.random.normal(0, 0.01, size=(N, 2))
        r = np.linalg.norm(pts, axis=1)
        mask = r > 0.99
        if np.any(mask):
            pts[mask] *= 0.99 / r[mask]
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    
    # Initial configuration
    pts = _hex_lattice_init(N)
    score = _compute_score(pts)
    
    # Force-directed optimization
    remaining = DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
    if remaining > 20:
        pts, score = _force_optimize(pts, time_budget=min(remaining - 30, 60))
    
    # Nelder-Mead polish
    remaining = DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
    if remaining > 5:
        pts, score = _nelder_mead_polish(pts, time_budget=min(remaining - 2, 30))
    
    return pts


# Test
if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    t1 = time.time()
    print(f"Time: {t1 - t0:.2f}s")
    print(f"Score: {_compute_score(pts)}")
    print(f"Shape: {pts.shape}")
