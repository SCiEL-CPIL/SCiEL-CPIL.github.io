"""Heilbronn convex regions: maximize min triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute all triangle index triplets for vectorized area computation
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]
_I, _J, _K = _TRIPLETS.T

def _triangle_areas_vectorized(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    # pts shape (N, 2)
    pi = pts[_I]  # (M, 2)
    pj = pts[_J]
    pk = pts[_K]
    # Cross product of (pj-pi) and (pk-pi)
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
    # Soft min using log-sum-exp trick
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
    
    # Generate diverse initial candidates
    candidates = []
    rng = np.random.default_rng(42)
    
    # 1. Regular 13-gon
    candidates.append(_regular_polygon(N, 1.0))
    
    # 2. Concentric rings (various splits)
    for n_outer in [7, 8, 9, 10, 11]:
        n_inner = N - n_outer
        if n_inner >= 3:
            for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
                candidates.append(_center_ring(n_outer, n_inner, 1.0, r_inner))
    
    # 3. Halton sequence points (low-discrepancy)
    halton = _halton_points(N)
    candidates.append(halton * 2)
    candidates.append(halton * 2 + 0.5)
    
    # 4. Angular parameterization: random angles with bounded radii
    for i in range(20):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = rng.uniform(0.5, 1.0, N)
        pts = _angular_parameterization(angles, radii)
        candidates.append(pts)
    
    # 5. Perturbed regular polygon
    for i in range(15):
        noise = rng.normal(0, 0.05, (N, 2))
        candidates.append(_regular_polygon(N, 1.0) + noise)
    
    # 6. Random points in disk
    for i in range(10):
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = rng.uniform(0, 1, N)
        pts = _angular_parameterization(angles, radii)
        candidates.append(pts)
    
    # Evaluate all candidates with exact score
    for pts in candidates:
        if time.time() > deadline:
            break
        s = _score(pts)
        update_best(pts, s)
    
    # Sort candidates by score, optimize top ones
    scored_candidates = []
    for pts in candidates:
        s = _score(pts)
        scored_candidates.append((s, pts))
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Use top candidates as starting points for optimization
    top_candidates = [pts for _, pts in scored_candidates[:min(15, len(scored_candidates))]]
    
    # Progressive beta: start smooth, get sharper
    betas = [20.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
        
        for init_pts in top_candidates:
            if time.time() > deadline - 15:
                break
            
            try:
                res = minimize(
                    _soft_min_objective,
                    init_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                res_pts = res.x.reshape(N, 2)
                res_score = _score(res_pts)
                update_best(res_pts, res_score)
                
            except Exception:
                pass
        
        # Update top candidates with current best
        if best_pts is not None:
            top_candidates = [best_pts.copy()] + top_candidates[:5]
    
    # Final refinement: use exact score with Nelder-Mead on the best point
    if best_pts is not None and time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 2000}
            )
            res_pts = res.x.reshape(N, 2)
            res_score = _score(res_pts)
            update_best(res_pts, res_score)
        except Exception:
            pass
    
    # One more round of L-BFGS from the best with high beta if time permits
    if best_pts is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                _soft_min_objective,
                best_pts.flatten(),
                args=(500.0,),
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
