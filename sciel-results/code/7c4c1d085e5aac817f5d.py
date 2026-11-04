import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Precompute all triangle index triplets for vectorized area computation
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]
_I, _J, _K = _TRIPLETS.T

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    a = pts[_I]
    b = pts[_J]
    c = pts[_K]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                          (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        if len(pts) < 3:
            return 0.0
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
    return float(np.min(areas)) / hull_area

def _soft_min_objective(x, beta=50.0):
    """
    Soft-min of triangle areas / hull area, negative for minimization.
    x is a flat array of length 2*N.
    """
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-10:
        return 1e6
    
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    
    # Soft min using log-sum-exp trick for numerical stability
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    return -soft_min_area / hull_area

def _exact_objective(x):
    """Exact negative score for Nelder-Mead refinement."""
    pts = x.reshape(-1, 2)
    return -_score(pts)

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _center_ring(n_outer, n_inner, r_outer=1.0, r_inner=0.5):
    """Points on two concentric rings."""
    angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + np.pi/n_inner
    outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    return np.vstack([outer, inner])

def _halton_sequence(n, base):
    """Generate first n Halton sequence values in [0,1)."""
    seq = np.zeros(n)
    f = 1.0
    for i in range(1, n+1):
        f /= base
        seq[i-1] = 0.0
        j = i
        while j > 0:
            seq[i-1] += (j % base) * f
            j //= base
    return seq

def _halton_points(n, base1=2, base2=3):
    """Generate n points using 2D Halton sequence."""
    x = _halton_sequence(n, base1)
    y = _halton_sequence(n, base2)
    return np.column_stack([x, y])

def _koksma_points(n):
    """Koksma-type arrangement: points on nested polygons."""
    pts = []
    # Start with a regular pentagon, then add points at golden-ratio radii
    # This is a heuristic inspired by optimal configurations
    r = 1.0
    for i in range(5):
        angle = 2 * np.pi * i / 5
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    
    golden = (1 + np.sqrt(5)) / 2
    r2 = 1.0 / golden
    for i in range(4):
        angle = 2 * np.pi * i / 4 + np.pi/4
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    r3 = r2 / golden
    for i in range(4):
        angle = 2 * np.pi * i / 4
        pts.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    return np.array(pts[:n])

def _fibonacci_lattice(n):
    """Fibonacci lattice points in unit disk."""
    pts = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        r = np.sqrt((i + 0.5) / n)
        theta = i * 2 * np.pi / golden_ratio
        pts.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    best_score = 0.0
    best_points = None
    
    def update_best(pts, score):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Regular 13-gon
    candidates.append(_regular_polygon(N, 1.0))
    
    # 2. Concentric rings (various splits)
    for n_outer in [7, 8, 9, 10]:
        n_inner = N - n_outer
        if n_inner >= 3:
            for r_inner in [0.4, 0.5, 0.6, 0.7]:
                candidates.append(_center_ring(n_outer, n_inner, 1.0, r_inner))
    
    # 3. Halton sequence points (low-discrepancy)
    halton = _halton_points(N)
    candidates.append(halton)
    candidates.append(halton * 2)
    # Also try scaled to unit disk
    radii = np.sqrt(halton[:, 0]**2 + halton[:, 1]**2)
    radii[radii < 1e-10] = 1.0
    candidates.append(halton / radii[:, np.newaxis])
    
    # 4. Random points on circle (sorted by angle for convex position)
    rng = np.random.default_rng(42)
    for i in range(15):
        angles = np.sort(rng.uniform(0, 2*np.pi, N))
        radii = rng.uniform(0.7, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # 5. Random points in disk
    for i in range(10):
        angles = rng.uniform(0, 2*np.pi, N)
        radii = rng.uniform(0, 1, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # 6. Perturbed regular polygon
    for i in range(10):
        noise = rng.normal(0, 0.05, (N, 2))
        candidates.append(_regular_polygon(N, 1.0) + noise)
    
    # 7. Koksma-type arrangement
    candidates.append(_koksma_points(N))
    
    # 8. Fibonacci lattice
    candidates.append(_fibonacci_lattice(N))
    
    # 9. Three-ring configuration
    for r1, r2, r3 in [(1.0, 0.7, 0.4), (1.0, 0.6, 0.3), (1.0, 0.8, 0.5)]:
        n1, n2, n3 = 5, 4, 4
        angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
        angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + np.pi/n2
        angles3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + np.pi/n3
        p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        p3 = np.column_stack([r3 * np.cos(angles3), r3 * np.sin(angles3)])
        candidates.append(np.vstack([p1, p2, p3]))
    
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
    top_candidates = [pts for _, pts in scored_candidates[:min(12, len(scored_candidates))]]
    
    # Progressive beta: start smooth, get sharper
    betas = [15.0, 30.0, 60.0, 120.0, 250.0]
    
    for beta in betas:
        if time.time() > deadline - 25:
            break
        
        for init_pts in top_candidates:
            if time.time() > deadline - 20:
                break
            
            try:
                res = minimize(
                    _soft_min_objective,
                    init_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
                )
                
                res_pts = res.x.reshape(N, 2)
                res_score = _score(res_pts)
                update_best(res_pts, res_score)
                
            except Exception:
                pass
        
        # Update top candidates with current best and a few others
        if best_points is not None:
            top_candidates = [best_points.copy()]
            # Add some diversity back
            for s, pts in scored_candidates[:3]:
                if not np.allclose(pts, best_points):
                    top_candidates.append(pts.copy())
    
    # Final refinement: use exact score with Nelder-Mead on the best point
    if best_points is not None and time.time() < deadline - 10:
        try:
            res = minimize(
                _exact_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 5000}
            )
            res_pts = res.x.reshape(N, 2)
            res_score = _score(res_pts)
            update_best(res_pts, res_score)
        except Exception:
            pass
    
    # Second refinement pass with even tighter tolerances
    if best_points is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                _exact_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 3000}
            )
            res_pts = res.x.reshape(N, 2)
            res_score = _score(res_pts)
            update_best(res_pts, res_score)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
        best_score = _score(best_points)
    
    return best_points
