"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 13

def _all_triangle_areas(pts):
    """Vectorized computation of all triangle areas."""
    n = len(pts)
    # Precompute all pairs
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    ax = pj[:, 0] - pi[:, 0]
    ay = pj[:, 1] - pi[:, 1]
    bx = pk[:, 0] - pi[:, 0]
    by = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(ax * by - ay * bx)
    return areas

def _min_triangle_area(pts):
    """Compute the minimum area over all triples of points."""
    areas = _all_triangle_areas(pts)
    return np.min(areas)

def _hull_area(pts):
    """Compute the convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_area = _hull_area(pts)
    if hull_area <= 0:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _soft_min_area(pts, alpha=50.0):
    """
    Compute a smooth approximation of the minimum triangle area.
    f(P) = (sum A_i^{-alpha})^{-1/alpha}
    """
    areas = _all_triangle_areas(pts)
    if np.any(areas < 1e-10):
        return 0.0
    sum_inv_pow = np.sum(areas ** (-alpha))
    if sum_inv_pow == 0:
        return 0.0
    return (sum_inv_pow ** (-1.0 / alpha))

def _neg_soft_score(params, alpha=50.0):
    """
    Negative soft score for minimization.
    """
    pts = params.reshape((N, 2))
    hull_area = _hull_area(pts)
    if hull_area < 1e-6:
        return 1e6
    
    soft_area = _soft_min_area(pts, alpha)
    if soft_area < 1e-6:
        return 1e6
    
    score = soft_area / hull_area
    return -score

def _neg_hard_score(params):
    """
    Negative exact score for minimization (non-smooth, for local refinement).
    """
    pts = params.reshape((N, 2))
    hull_area = _hull_area(pts)
    if hull_area < 1e-6:
        return 1e6
    min_area = _min_triangle_area(pts)
    if min_area < 1e-10:
        return 1e6
    score = min_area / hull_area
    return -score

def _parametrize_ring_center(ring_angles, ring_radii, center_offset):
    """
    Parameterize 13 points: 12 ring points + 1 center point.
    ring_angles: array of 12 angles (radians)
    ring_radii: array of 12 radii
    center_offset: (2,) offset for center point
    Returns: (13, 2) array of points
    """
    ring_pts = np.column_stack([
        ring_radii * np.cos(ring_angles),
        ring_radii * np.sin(ring_angles)
    ])
    center_pt = np.array([center_offset[0], center_offset[1]])
    return np.vstack([ring_pts, center_pt])

def _unpack_ring_center_params(params):
    """
    Unpack flat parameter vector into ring_angles, ring_radii, center_offset.
    params layout: [12 angles, 12 radii, 2 center offset] = 26 params
    """
    ring_angles = params[0:12]
    ring_radii = params[12:24]
    center_offset = params[24:26]
    return ring_angles, ring_radii, center_offset

def _neg_soft_score_ring_center(params, alpha=50.0):
    """
    Soft score with ring+center parameterization (14 effective DOF via 26 params,
    but angles/radii are naturally bounded).
    """
    ring_angles, ring_radii, center_offset = _unpack_ring_center_params(params)
    pts = _parametrize_ring_center(ring_angles, ring_radii, center_offset)
    return _neg_soft_score(pts, alpha)

def _neg_hard_score_ring_center(params):
    """
    Hard score with ring+center parameterization.
    """
    ring_angles, ring_radii, center_offset = _unpack_ring_center_params(params)
    pts = _parametrize_ring_center(ring_angles, ring_radii, center_offset)
    return _neg_hard_score(pts)

def _make_initial_guesses():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts1 = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(pts1)
    
    # 2. Regular 12-gon + Center
    angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    outer = np.column_stack([np.cos(angles12), np.sin(angles12)])
    center = np.array([[0.0, 0.0]])
    pts2 = np.vstack([outer, center])
    configs.append(pts2)
    
    # 3. Two rings: 7 outer, 6 inner
    angles7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles6 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/7
    outer = np.column_stack([np.cos(angles7), np.sin(angles7)])
    inner = np.column_stack([np.cos(angles6)*0.5, np.sin(angles6)*0.5])
    pts3 = np.vstack([outer, inner])
    configs.append(pts3)
    
    # 4. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.uniform(-1, 1, (N, 2))
        configs.append(pts)
        
    # 5. Grid-like
    grid_pts = []
    for i in range(4):
        for j in range(4):
            if i == 3 and j == 3:
                continue
            grid_pts.append([i - 1.5, j - 1.5])
    pts_grid = np.array(grid_pts)
    configs.append(pts_grid)
    
    # 6. Ring + center with slight perturbations
    for _ in range(3):
        angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False) + rng.normal(0, 0.1, 12)
        angles12 = np.sort(angles12)
        radii = 1.0 + rng.normal(0, 0.1, 12)
        radii = np.clip(radii, 0.5, 1.5)
        ring_pts = np.column_stack([radii * np.cos(angles12), radii * np.sin(angles12)])
        center = rng.normal(0, 0.1, 2).reshape(1, 2)
        pts6 = np.vstack([ring_pts, center])
        configs.append(pts6)
            
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_pts = None
    
    initial_guesses = _make_initial_guesses()
    
    # Evaluate initial guesses
    for guess in initial_guesses:
        score = _score_points(guess)
        if score > best_score:
            best_score = score
            best_pts = guess.copy()
            
    if time.time() > deadline:
        return best_pts if best_pts is not None else initial_guesses[0]
    
    # Phase 1: Differential Evolution on ring+center parameterization
    # This exploits the topological structure: 12 ring + 1 center
    # Bounds: angles in [0, 2pi], radii in [0.3, 2.0], center in [-1, 1]
    bounds = [(0, 2 * np.pi)] * 12 + [(0.3, 2.0)] * 12 + [(-1.0, 1.0)] * 2
    
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                _neg_soft_score_ring_center,
                bounds=bounds,
                args=(30.0,),
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,
                init='sobol'
            )
            if de_result.fun < -best_score:
                candidate_pts = _parametrize_ring_center(*_unpack_ring_center_params(de_result.x))
                candidate_score = _score_points(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement from best solution using Nelder-Mead
    if time.time() < deadline - 30:
        # Convert best_pts to ring+center params if it looks like a ring+center config
        # Otherwise, just use full 2D params
        # Try both: full 2D and ring+center
        
        # 2a: Full 2D Nelder-Mead refinement
        starts = [best_pts.copy()]
        rng = np.random.default_rng(123)
        for _ in range(3):
            noise = rng.normal(0, 0.05, size=(N, 2))
            starts.append(best_pts + noise)
        
        for start_pts in starts:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    _neg_soft_score,
                    start_pts.flatten(),
                    args=(50.0,),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                candidate_pts = res.x.reshape((N, 2))
                candidate_score = _score_points(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts.copy()
            except Exception:
                pass
        
        # 2b: Ring+center Nelder-Mead refinement
        if time.time() < deadline - 20:
            # Extract ring+center params from best_pts
            # Sort by distance from centroid to identify ring vs center
            centroid = np.mean(best_pts, axis=0)
            dists = np.linalg.norm(best_pts - centroid, axis=1)
            # Assume the point closest to centroid is the center
            center_idx = np.argmin(dists)
            ring_idx = [i for i in range(N) if i != center_idx]
            
            ring_pts = best_pts[ring_idx]
            center_pt = best_pts[center_idx]
            
            # Compute angles and radii
            ring_angles = np.arctan2(ring_pts[:, 1], ring_pts[:, 0])
            ring_angles = np.sort(ring_angles)
            ring_radii = np.linalg.norm(ring_pts, axis=1)
            center_offset = center_pt - centroid  # relative to centroid
            
            # But we need absolute center offset, not relative
            # Actually, the parameterization uses absolute center position
            # So center_offset should be the actual center position
            center_offset_abs = center_pt.copy()
            
            init_params = np.concatenate([ring_angles, ring_radii, center_offset_abs])
            
            # Ensure angles are in [0, 2pi]
            ring_angles_mod = ring_angles % (2 * np.pi)
            ring_angles_mod = np.sort(ring_angles_mod)
            init_params = np.concatenate([ring_angles_mod, ring_radii, center_offset_abs])
            
            # Bounds for ring+center
            bounds_rc = [(0, 2 * np.pi)] * 12 + [(0.3, 2.0)] * 12 + [(-1.0, 1.0)] * 2
            
            try:
                res = minimize(
                    _neg_soft_score_ring_center,
                    init_params,
                    args=(50.0,),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                candidate_pts = _parametrize_ring_center(*_unpack_ring_center_params(res.x))
                candidate_score = _score_points(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts.copy()
            except Exception:
                pass
    
    # Phase 3: Final hard-score refinement with Nelder-Mead
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _neg_hard_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            candidate_pts = res.x.reshape((N, 2))
            candidate_score = _score_points(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()
        except Exception:
            pass
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(_score_points(pts))
