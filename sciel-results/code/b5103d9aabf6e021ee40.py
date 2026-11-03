import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Precompute all triangle index triples
_TRIS = np.array(list(combinations(range(N), 3)))
_N_TRIS = _TRIS.shape[0]

def _triangle_areas(pts):
    """Vectorized computation of all C(N,3) triangle areas."""
    i, j, k = _TRIS[:, 0], _TRIS[:, 1], _TRIS[:, 2]
    p1 = pts[i]
    p2 = pts[j]
    p3 = pts[k]
    # 2*area = |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_triangle_area(pts):
    return np.min(_triangle_areas(pts))

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    min_area = _min_triangle_area(pts)
    h_area = _hull_area(pts)
    if h_area < 1e-10:
        return 0.0
    return min_area / h_area

def _soft_min_score(x, beta):
    """Smooth surrogate: soft-min of triangle areas divided by hull area.
    Lower beta -> sharper approximation to true min."""
    pts = x.reshape(N, 2)
    areas = _triangle_areas(pts)
    # Soft minimum: exp(-beta * min) approximated by log-sum-exp trick
    # soft_min = -log(sum(exp(-beta * areas)))/beta
    # For numerical stability:
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = min_a - log_sum / beta
    h_area = _hull_area(pts)
    if h_area < 1e-10:
        return 0.0
    return soft_min_val / h_area

def _neg_soft_min_score(x, beta):
    return -_soft_min_score(x, beta)

def _neg_score(x):
    pts = x.reshape(N, 2)
    return -_score(pts)

def _regular_polygon(n=N, noise=0.0, rng=None):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    if noise > 0 and rng is not None:
        pts = pts + rng.normal(0, noise, (n, 2))
    return pts

def _center_ring(n=N, r_inner=0.5, r_outer=1.0, rng=None):
    """Points on concentric rings."""
    pts = np.zeros((n, 2))
    for i in range(n):
        angle = 2 * np.pi * i / n
        r = r_inner + (r_outer - r_inner) * (i % 2)
        pts[i, 0] = r * np.cos(angle)
        pts[i, 1] = r * np.sin(angle)
    if rng is not None:
        pts = pts + rng.normal(0, 0.05, (n, 2))
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    rng = np.random.default_rng(123)
    
    best_score = 0.0
    best_points = None
    
    def _update_best(pts):
        nonlocal best_score, best_points
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        return s
    
    # Generate diverse initial candidates
    candidates = []
    
    # Regular 13-gon
    candidates.append(_regular_polygon())
    
    # Perturbed regular polygons with varying noise
    for noise in [0.05, 0.1, 0.2, 0.3, 0.5]:
        candidates.append(_regular_polygon(noise=noise, rng=rng))
    
    # Center-ring configurations
    for r_inner in [0.3, 0.5, 0.7]:
        candidates.append(_center_ring(r_inner=r_inner, rng=rng))
    
    # Random points on unit circle (sorted angles)
    for _ in range(8):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        candidates.append(pts)
    
    # Random points in unit disk
    for _ in range(8):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        candidates.append(pts)
    
    # Evaluate all initial candidates
    for cand in candidates:
        _update_best(cand)
    
    # Phase 1: L-BFGS-B local optimization with soft-min surrogate from each candidate
    # Use increasing beta (sharper) for refinement
    betas = [20, 50, 100, 200]
    
    for cand in candidates:
        if time.time() > deadline:
            break
        
        x0 = cand.flatten()
        current_best = x0.copy()
        current_score = _score(cand)
        
        for beta in betas:
            if time.time() > deadline:
                break
            try:
                res = minimize(_neg_soft_min_score, current_best, args=(beta,),
                              method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                res_pts = res.x.reshape(N, 2)
                res_score = _score(res_pts)
                if res_score > current_score:
                    current_best = res.x
                    current_score = res_score
            except Exception:
                break
        
        _update_best(current_best.reshape(N, 2))
    
    # Phase 2: Differential Evolution for global search
    if time.time() < deadline - 30:
        try:
            bounds = [(-1.5, 1.5)] * (N * 2)
            
            def de_objective(x):
                return _neg_soft_min_score(x, beta=50)
            
            de_res = differential_evolution(
                de_objective, bounds,
                seed=42, maxiter=100, popsize=15,
                tol=1e-6, polish=True,
                init='halton',
                updating='immediate'
            )
            
            de_pts = de_res.x.reshape(N, 2)
            _update_best(de_pts)
            
            # Polish DE result with L-BFGS-B
            if time.time() < deadline - 10:
                for beta in [100, 200, 500]:
                    try:
                        res = minimize(_neg_soft_min_score, de_res.x, args=(beta,),
                                      method='L-BFGS-B',
                                      options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                        res_pts = res.x.reshape(N, 2)
                        _update_best(res_pts)
                        de_res = res
                    except Exception:
                        break
        except Exception:
            pass
    
    # Phase 3: Final polish of best point found
    if best_points is not None and time.time() < deadline - 5:
        x_best = best_points.flatten()
        for beta in [200, 500, 1000]:
            try:
                res = minimize(_neg_soft_min_score, x_best, args=(beta,),
                              method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12})
                res_pts = res.x.reshape(N, 2)
                _update_best(res_pts)
                x_best = res.x
            except Exception:
                break
    
    # Fallback
    if best_points is None:
        best_points = _regular_polygon()
        best_score = _score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
