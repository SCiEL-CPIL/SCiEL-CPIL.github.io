"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    i, j, k = _TRIPLETS.T
    a = pts[i]
    b = pts[j]
    c = pts[k]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas) / hull_area

def _soft_min_objective(x, beta=50.0):
    """Negative soft-min of (min triangle area / hull area) for minimization.
    x: (2N,) flattened point coordinates."""
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 1e6
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    scaled = -areas * beta
    max_scaled = np.max(scaled)
    soft_min = -max_scaled / beta - np.log(np.sum(np.exp(scaled - max_scaled))) / beta
    return -soft_min / hull_area

def _exact_objective(x):
    """Negative exact score for minimization (non-smooth)."""
    pts = x.reshape(-1, 2)
    return -_score(pts)

def _halton_sequence(n, base):
    """Generate first n Halton sequence values with given base."""
    result = []
    for i in range(1, n + 1):
        f = 1.0
        r = 0.0
        idx = i
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        result.append(r)
    return np.array(result)

def _halton_points_2d(n, seed=0):
    """Generate n 2D points using Halton sequences (bases 2 and 3)."""
    rng = np.random.RandomState(seed)
    offset = rng.randint(1, 1000)
    h2 = _halton_sequence(n + offset, 2)
    h3 = _halton_sequence(n + offset, 3)
    pts = np.column_stack([h2[offset:offset + n], h3[offset:offset + n]])
    return pts

def _concentric_ring_init(seed=0, n_inner=5, n_outer=8):
    """Generate points on two concentric rings for diverse initialization."""
    rng = np.random.RandomState(seed)
    pts = np.zeros((N, 2))
    # Inner ring
    angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + rng.uniform(0, 2*np.pi)
    r_in = 0.4 + rng.uniform(-0.1, 0.1)
    pts[:n_inner, 0] = r_in * np.cos(angles_in)
    pts[:n_inner, 1] = r_in * np.sin(angles_in)
    # Outer ring
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + rng.uniform(0, 2*np.pi)
    r_out = 1.0 + rng.uniform(-0.1, 0.1)
    pts[n_inner:, 0] = r_out * np.cos(angles_out)
    pts[n_inner:, 1] = r_out * np.sin(angles_out)
    return pts

def _regular_polygon_init(seed=0):
    """Regular polygon with small perturbation."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 2*np.pi)
    r = 1.0 + rng.normal(0, 0.05, N)
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    return pts

def _nelder_mead_polish(pts, max_iter=500, tol=1e-10):
    """Polish using Nelder-Mead on the exact (non-smooth) objective."""
    x0 = pts.flatten()
    try:
        result = minimize(
            _exact_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return pts

def _elliptic_regular_init(a, b, c, angle):
    """Apply elliptical distortion to a regular 13-gon.
    Parameters a, b, c define the symmetric matrix [[a, c/2], [c/2, b]].
    """
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    # Rotate by angle
    theta = angle
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    base_pts = base_pts @ rot.T
    # Apply elliptical transform
    M = np.array([[a, c / 2], [c / 2, b]])
    pts = base_pts @ M.T
    return pts

def _elliptic_objective(params):
    """Objective for elliptic parameter optimization.
    params = [a, b, c, angle]
    """
    a, b, c, angle = params
    # Ensure positive definiteness
    if a <= 0 or b <= 0 or a * b - (c / 2) ** 2 <= 0:
        return 1e6
    try:
        pts = _elliptic_regular_init(a, b, c, angle)
    except Exception:
        return 1e6
    return _soft_min_objective(pts.flatten(), beta=100.0)

def _optimize_elliptic(start_params, max_iter=200):
    """Optimize elliptic parameters."""
    try:
        result = minimize(
            _elliptic_objective,
            start_params,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result.success:
            a, b, c, angle = result.x
            if a > 0 and b > 0 and a * b - (c / 2) ** 2 > 0:
                pts = _elliptic_regular_init(a, b, c, angle)
                return pts
    except Exception:
        pass
    return None

def _radial_to_pts(params):
    """Convert radial parameters (angle, radius) to (N, 2) points.
    params: (2N,) = [angle_0, r_0, angle_1, r_1, ...]
    Angles are sorted to ensure convex ordering."""
    angles = params[0::2]
    radii = params[1::2]
    # Sort by angle to ensure proper ordering
    sorted_idx = np.argsort(angles)
    angles = angles[sorted_idx]
    radii = radii[sorted_idx]
    # Normalize angles to [0, 2pi)
    angles = np.mod(angles, 2 * np.pi)
    # Ensure radii are positive
    radii = np.maximum(radii, 0.01)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _soft_min_radial_objective(params, beta=50.0):
    """Soft-min objective in radial parameterization."""
    pts = _radial_to_pts(params)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 1e6
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    scaled = -areas * beta
    max_scaled = np.max(scaled)
    soft_min = -max_scaled / beta - np.log(np.sum(np.exp(scaled - max_scaled))) / beta
    return -soft_min / hull_area

def _radial_init_from_pts(pts, seed=0):
    """Initialize radial parameters from existing points."""
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    # Sort by angle
    sorted_idx = np.argsort(angles)
    angles = angles[sorted_idx]
    radii = radii[sorted_idx]
    params = np.zeros(2 * N)
    params[0::2] = angles
    params[1::2] = radii
    return params

def _radial_init_regular(seed=0):
    """Regular polygon in radial parameterization."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 2*np.pi)
    radii = 1.0 + rng.normal(0, 0.05, N)
    params = np.zeros(2 * N)
    params[0::2] = angles
    params[1::2] = radii
    return params

def _radial_init_concentric(seed=0, n_inner=5, n_outer=8):
    """Concentric rings in radial parameterization."""
    rng = np.random.RandomState(seed)
    params = np.zeros(2 * N)
    # Inner ring
    angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + rng.uniform(0, 2*np.pi)
    r_in = 0.4 + rng.uniform(-0.1, 0.1)
    params[0:2*n_inner:2] = angles_in
    params[1:2*n_inner:2] = r_in
    # Outer ring
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + rng.uniform(0, 2*np.pi)
    r_out = 1.0 + rng.uniform(-0.1, 0.1)
    params[2*n_inner:2*N:2] = angles_out
    params[2*n_inner+1:2*N:2] = r_out
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_score = 0.0
    best_pts = _regular_polygon_init(0)
    
    # Phase 1: Multi-start with soft-min L-BFGS-B on full 2D coordinates
    init_strategies = [
        _regular_polygon_init,
        _concentric_ring_init,
        lambda s: _halton_points_2d(N, seed=s) * 0.8,
    ]
    
    n_starts = 25
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        strategy_idx = i % len(init_strategies)
        seed = i
        try:
            pts0 = init_strategies[strategy_idx](seed)
            x0 = pts0.flatten()
            
            result = minimize(
                _soft_min_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            pts = result.x.reshape(-1, 2)
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Phase 2: Elliptic parameter optimization for structured solutions
    if time.time() < deadline - 15:
        # Try several elliptic parameter starts
        elliptic_starts = [
            [1.0, 1.0, 0.0, 0.0],
            [1.2, 0.8, 0.0, 0.0],
            [0.8, 1.2, 0.0, 0.0],
            [1.0, 1.0, 0.3, 0.0],
            [1.1, 0.9, -0.2, np.pi/6],
            [0.9, 1.1, 0.2, -np.pi/6],
        ]
        for sp in elliptic_starts:
            if time.time() > deadline - 10:
                break
            try:
                pts = _optimize_elliptic(sp, max_iter=150)
                if pts is not None:
                    score = _score(pts)
                    if score > best_score:
                        best_score = score
                        best_pts = pts.copy()
            except Exception:
                continue
    
    # Phase 3: Radial parameterization optimization
    if time.time() < deadline - 20:
        radial_strategies = [
            _radial_init_regular,
            _radial_init_concentric,
        ]
        
        n_radial_starts = 15
        for i in range(n_radial_starts):
            if time.time() > deadline - 15:
                break
            
            strategy_idx = i % len(radial_strategies)
            seed = i
            try:
                params0 = radial_strategies[strategy_idx](seed)
                
                result = minimize(
                    _soft_min_radial_objective,
                    params0,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                params = result.x
                pts = _radial_to_pts(params)
                score = _score(pts)
                
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
            except Exception:
                continue
    
    # Phase 4: Nelder-Mead polish on the best solution
    if time.time() < deadline - 10:
        polished = _nelder_mead_polish(best_pts, max_iter=500)
        score = _score(polished)
        if score > best_score:
            best_score = score
            best_pts = polished.copy()
    
    # Phase 5: A few more diverse starts with full polish if time permits
    for seed in range(5):
        if time.time() > deadline - 5:
            break
        
        try:
            pts0 = _concentric_ring_init(seed + 100)
            x0 = pts0.flatten()
            
            result = minimize(
                _soft_min_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-8}
            )
            
            pts = result.x.reshape(-1, 2)
            # Quick NM polish
            pts = _nelder_mead_polish(pts, max_iter=200)
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    return best_pts
# EVOLVE-BLOCK-END
