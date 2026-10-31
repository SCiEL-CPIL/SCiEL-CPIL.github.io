import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 16
D = 2

def _pairwise_distances(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))

def _score(pts):
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    dmax = d.max()
    if dmin <= 1e-12:
        return np.inf
    return (dmax / dmin) ** 2

def _objective(pts_flat):
    pts = pts_flat.reshape(N, 2)
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    dmax = d.max()
    if dmin <= 1e-12:
        return 1e10
    return (dmax / dmin) ** 2

def generate_hex_cluster(n_points=19):
    """Generate the 19-point hexagonal cluster (center + 2 shells)."""
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    
    pts = []
    # Axial coordinates (q, r)
    # We need to cover the 19 points:
    # Center (0,0)
    # 1st shell: 6 points
    # 2nd shell: 12 points
    # Total 19.
    
    # Generate all points with dist <= 2 in axial coords
    # Axial dist: sqrt(q^2 + r^2 + qr)
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Check if in hexagon of radius 2
            # Condition: max(|q|, |r|, |q+r|) <= 2 ?
            # Or simply check distance
            d_axial = np.sqrt(q*q + r*r + q*r)
            if d_axial <= 2 + 1e-6:
                p = q * v1 + r * v2
                pts.append(p)
                
    pts = np.array(pts)
    # Center it
    pts -= pts.mean(axis=0)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_pts = None
    best_score = np.inf
    
    # 1. Generate 19-point hex cluster
    hex_19 = generate_hex_cluster(19)
    
    # Verify it has 19 points
    if len(hex_19) != 19:
        # Fallback to simple grid if generation fails
        x, y = np.meshgrid(np.arange(4), np.arange(4))
        hex_19 = np.column_stack([x.ravel(), y.ravel()]).astype(float)
        hex_19 -= hex_19.mean(axis=0)
        
    # 2. Try all subsets of 16 points from the 19
    # This is C(19, 3) = 969 combinations
    # We remove 3 points
    indices = np.arange(19)
    best_subset_score = np.inf
    best_subset_pts = None
    
    for removed_indices in itertools.combinations(indices, 3):
        mask = np.ones(19, dtype=bool)
        mask[list(removed_indices)] = False
        subset = hex_19[mask]
        
        # Quick check: ensure distinct (they are)
        s = _score(subset)
        if s < best_subset_score:
            best_subset_score = s
            best_subset_pts = subset.copy()
            
    if best_subset_pts is not None:
        best_pts = best_subset_pts
        best_score = best_subset_score
        
    # 3. Local Optimization
    # Use Powell on the best discrete candidate
    if best_pts is not None and time.time() < deadline - 20:
        try:
            result = minimize(
                _objective,
                best_pts.ravel(),
                method='Powell',
                options={'maxiter': 2000, 'maxfev': 50000, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            if result.fun < best_score:
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
        except Exception:
            pass
            
    # 4. Refinement with Nelder-Mead
    if best_pts is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                _objective,
                best_pts.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_score:
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
        except Exception:
            pass
            
    # 5. Fallback
    if best_pts is None:
        return make_hexagonal_cluster(N)
        
    return best_pts

def make_hexagonal_cluster(n, spread=5):
    """Fallback generator."""
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    pts = []
    for i in range(-spread, spread + 1):
        for j in range(-spread, spread + 1):
            p = i * v1 + j * v2
            pts.append(p)
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    return pts[order[:n]]
