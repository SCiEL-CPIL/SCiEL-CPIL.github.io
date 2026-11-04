"""Heilbronn convex regions: maximize min triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute all triangle index triplets for vectorized area computation
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]
_I, _J, _K = _TRIPLETS.T

def _triangle_areas_vectorized(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
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
    if hull_area <= 1e-15:
        return 0.0
    areas = _triangle_areas_vectorized(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas) / hull_area

def _soft_min_objective(x, beta=50.0):
    """Soft-min of triangle areas / hull area, negative for minimization."""
    pts = x.reshape(N, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-15:
        return 1e6
    areas = _triangle_areas_vectorized(pts)
    if len(areas) == 0:
        return 1e6
    scaled = -areas * beta
    max_scaled = np.max(scaled)
    soft_min = -max_scaled / beta - np.log(np.sum(np.exp(scaled - max_scaled))) / beta
    return -soft_min / hull_area

def _regular_polygon(n, radius=1.0):
    """Generate regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _center_ring(n_outer, n_inner, r_outer=1.0, r_inner=0.5):
    """Points on two concentric rings."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    return np.vstack([outer, inner])

def _halton_sequence(n, base):
    """Generate first n Halton sequence values in [0,1)."""
    seq = np.zeros(n)
    f = 1.0
    for i in range(1, n + 1):
        f /= base
        seq[i - 1] = 0.0
        j = i
        while j > 0:
            seq[i - 1] += (j % base) * f
            j //= base
    return seq

def _halton_points(n, base1=2, base2=3):
    """Generate n points using 2D Halton sequence."""
    x = _halton_sequence(n, base1)
    y = _halton_sequence(n, base2)
    return np.column_stack([x, y])

def _angular_parameterization(angles, radii):
    """Convert angles and radii to Cartesian points."""
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _convex_polygon_from_params(angles, radii):
    """
    Create a convex polygon from sorted angles and radii.
    angles: sorted array of N angles in [0, 2*pi)
    radii: array of N positive radii
    Returns: (N, 2) array of points forming a convex polygon
    """
    # Sort by angle to ensure proper ordering
    order = np.argsort(angles)
    sorted_angles = angles[order]
    sorted_radii = radii[order]
    
    # Normalize angles to be in [0, 2*pi) and ensure they're spread
    sorted_angles = sorted_angles % (2 * np.pi)
    
    # Ensure minimum angular separation
    min_sep = 0.1
    for i in range(N):
        if i == 0:
            continue
        if sorted_angles[i] - sorted_angles[i-1] < min_sep:
            sorted_angles[i] = sorted_angles[i-1] + min_sep
    
    # Wrap around
    if sorted_angles[-1] + 2*np.pi - sorted_angles[0] < min_sep:
        # Redistribute
        sorted_angles = np.linspace(0, 2*np.pi, N, endpoint=False) + np.random.default_rng(0).uniform(0, 0.1, N)
        sorted_angles = np.sort(sorted_angles)
    
    return np.column_stack([sorted_radii * np.cos(sorted_angles), sorted_radii * np.sin(sorted_angles)])

def _objective_from_params(params, beta=50.0):
    """
    Objective function for convex polygon parameterization.
    params: (2N,) array - first N are angles, last N are log(radii)
    """
    angles = params[:N]
    log_radii = params[N:]
    
    # Ensure radii are positive
    radii = np.exp(log_radii)
    
    # Create convex polygon
    pts = _convex_polygon_from_params(angles, radii)
    
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-15:
        return 1e6
    
    areas = _triangle_areas_vectorized(pts)
    if len(areas) == 0:
        return 1e6
    
    # Soft min
    scaled = -areas * beta
    max_scaled = np.max(scaled)
    soft_min = -max_scaled / beta - np.log(np.sum(np.exp(scaled - max_scaled))) / beta
    return -soft_min / hull_area

def _exact_objective_from_params(params):
    """Exact objective for convex polygon parameterization."""
    angles = params[:N]
    log_radii = params[N:]
    radii = np.exp(log_radii)
    pts = _convex_polygon_from_params(angles, radii)
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_score = 0.0
    best_pts = None
    
    def update_best(pts, score):
        nonlocal best_score, best_pts
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates in Cartesian space
    candidates = []
    
    # 1. Regular 13-gon
    candidates.append(_regular_polygon(N, 1.0))
    
    # 2. Concentric rings
    for n_outer in [7, 8, 9, 10, 11]:
        n_inner = N - n_outer
        if n_inner >= 3:
            for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
                candidates.append(_center_ring(n_outer, n_inner, 1.0, r_inner))
    
    # 3. Halton sequence points
    halton = _halton_points(N)
    candidates.append(halton * 2)
    candidates.append(halton * 2 + 0.5)
    
    # 4. Random points in disk
    for i in range(15):
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = rng.uniform(0.3, 1.0, N)
        pts = _angular_parameterization(angles, radii)
        candidates.append(pts)
    
    # 5. Perturbed regular polygon
    for i in range(10):
        noise = rng.normal(0, 0.05, (N, 2))
        candidates.append(_regular_polygon(N, 1.0) + noise)
    
    # Evaluate all candidates
    for pts in candidates:
        if time.time() > deadline:
            break
        s = _score(pts)
        update_best(pts, s)
    
    # Convert best candidate to parameter space for convex polygon optimization
    def pts_to_params(pts):
        """Convert Cartesian points to (angles, log_radii) parameterization."""
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        angles = np.arctan2(dy, dx) % (2 * np.pi)
        radii = np.sqrt(dx**2 + dy**2)
        radii = np.maximum(radii, 0.01)  # Ensure positive
        log_radii = np.log(radii)
        return np.concatenate([angles, log_radii])
    
    # Get top candidates sorted by score
    scored_candidates = []
    for pts in candidates:
        s = _score(pts)
        scored_candidates.append((s, pts))
    scored_candidates.sort(key=lambda x: -x[0])
    
    top_candidates = [pts for _, pts in scored_candidates[:min(10, len(scored_candidates))]]
    
    # Phase 1: Differential evolution in parameter space (global search)
    if time.time() < deadline - 30:
        bounds = [(0, 2*np.pi)] * N + [(-2, 1)] * N  # angles in [0,2pi), log_radii in [-2,1]
        
        try:
            de_result = differential_evolution(
                _objective_from_params,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                atol=1e-8,
                polish=True,
                args=(50.0,)
            )
            
            if de_result.success:
                de_params = de_result.x
                de_pts = _convex_polygon_from_params(de_params[:N], np.exp(de_params[N:]))
                de_score = _score(de_pts)
                update_best(de_pts, de_score)
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B refinement from multiple starting points
    betas = [50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
        
        # Start from best and top candidates
        start_params_list = []
        
        if best_pts is not None:
            start_params_list.append(pts_to_params(best_pts))
        
        for pts in top_candidates[:5]:
            start_params_list.append(pts_to_params(pts))
        
        # Add some random starts
        for i in range(5):
            random_angles = np.sort(rng.uniform(0, 2*np.pi, N))
            random_log_radii = rng.uniform(-1, 0.5, N)
            start_params_list.append(np.concatenate([random_angles, random_log_radii]))
        
        for init_params in start_params_list:
            if time.time() > deadline - 15:
                break
            
            try:
                res = minimize(
                    _objective_from_params,
                    init_params,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(0, 2*np.pi)] * N + [(-2, 1)] * N,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                res_params = res.x
                res_pts = _convex_polygon_from_params(res_params[:N], np.exp(res_params[N:]))
                res_score = _score(res_pts)
                update_best(res_pts, res_score)
                
            except Exception:
                pass
        
        # Update top candidates
        if best_pts is not None:
            top_candidates = [best_pts.copy()] + top_candidates[:3]
    
    # Phase 3: Fine-tuning with higher beta
    if best_pts is not None and time.time() < deadline - 10:
        try:
            init_params = pts_to_params(best_pts)
            res = minimize(
                _objective_from_params,
                init_params,
                args=(500.0,),
                method='L-BFGS-B',
                bounds=[(0, 2*np.pi)] * N + [(-2, 1)] * N,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            res_params = res.x
            res_pts = _convex_polygon_from_params(res_params[:N], np.exp(res_params[N:]))
            res_score = _score(res_pts)
            update_best(res_pts, res_score)
        except Exception:
            pass
    
    # Phase 4: Try direct Cartesian optimization from best as fallback
    if best_pts is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                _soft_min_objective,
                best_pts.flatten(),
                args=(300.0,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            res_pts = res.x.reshape(N, 2)
            res_score = _score(res_pts)
            update_best(res_pts, res_score)
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _regular_polygon(N, 1.0)
    
    return best_pts
# EVOLVE-BLOCK-END
