import numpy as np
import time
from scipy.optimize import minimize
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
    """Smooth surrogate: soft-min of triangle areas divided by hull area."""
    pts = x.reshape(N, 2)
    areas = _triangle_areas(pts)
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

def _regular_polygon(n=N, noise=0.0, rng=None, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    if noise > 0 and rng is not None:
        pts = pts + rng.normal(0, noise, (n, 2))
    return pts

def _concentric_rings(n_outer, n_inner, r_outer=1.0, r_inner=0.5, offset_inner=True, rng=None, noise=0.0):
    """Points on two concentric rings."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    if offset_inner:
        angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    else:
        angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
    
    outer = r_outer * np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner = r_inner * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    pts = np.vstack([outer, inner])
    
    if noise > 0 and rng is not None:
        pts = pts + rng.normal(0, noise, (N, 2))
        
    return pts

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

def _halton_points(n, base1=2, base2=3, scale=1.0, offset=0.0):
    """Generate n points using 2D Halton sequence, scaled and centered."""
    x = _halton_sequence(n, base1)
    y = _halton_sequence(n, base2)
    pts = np.column_stack([x, y])
    # Center and scale to roughly unit disk
    pts = (pts - 0.5) * 2.0 * scale
    return pts + offset

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
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
    
    # 1. Regular 13-gon
    candidates.append(_regular_polygon())
    
    # 2. Concentric rings (various splits) - Key structural idea from inspiration
    splits = [(7, 6), (8, 5), (9, 4), (10, 3), (6, 7), (5, 8)]
    for n_out, n_in in splits:
        if n_in >= 3:
            for r_in in [0.4, 0.5, 0.6, 0.7]:
                candidates.append(_concentric_rings(n_out, n_in, 1.0, r_in, offset_inner=True))
                candidates.append(_concentric_rings(n_out, n_in, 1.0, r_in, offset_inner=False))
    
    # 3. Halton sequence points (low-discrepancy)
    for scale in [0.8, 1.0, 1.2]:
        candidates.append(_halton_points(N, 2, 3, scale=scale))
        candidates.append(_halton_points(N, 3, 5, scale=scale))
    
    # 4. Random points on circle (sorted angles)
    for _ in range(5):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        candidates.append(pts)
    
    # 5. Perturbed regular polygons
    for noise in [0.05, 0.1, 0.2]:
        candidates.append(_regular_polygon(noise=noise, rng=rng))
    
    # Evaluate all initial candidates
    scored_candidates = []
    for cand in candidates:
        if time.time() > deadline:
            break
        s = _score(cand)
        scored_candidates.append((s, cand))
        if s > best_score:
            best_score = s
            best_points = cand.copy()
    
    # Sort by score to prioritize good starts
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Take top unique candidates for optimization
    # Avoid too many similar starts
    top_candidates = []
    seen = []
    for s, pts in scored_candidates:
        # Check if similar to existing top candidates
        is_similar = False
        for existing in seen:
            if np.linalg.norm(pts - existing) < 0.1:
                is_similar = True
                break
        if not is_similar and len(top_candidates) < 8:
            top_candidates.append(pts)
            seen.append(pts)
    
    # Phase 1: L-BFGS-B local optimization with soft-min surrogate
    # Progressive beta: start smooth, get sharper
    betas = [20.0, 50.0, 100.0, 200.0]
    
    current_top = top_candidates[:]
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        new_top = []
        for init_pts in current_top:
            if time.time() > deadline:
                break
            
            try:
                res = minimize(
                    _neg_soft_min_score,
                    init_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                res_pts = res.x.reshape(N, 2)
                res_score = _score(res_pts)
                
                if res_score > best_score:
                    best_score = res_score
                    best_points = res_pts.copy()
                
                new_top.append(res_pts)
                
            except Exception:
                pass
        
        # Update top candidates with results, keep best few
        # Sort new_top by score
        scored_new = []
        for pts in new_top:
            s = _score(pts)
            scored_new.append((s, pts))
        scored_new.sort(key=lambda x: -x[0])
        
        current_top = [pts for _, pts in scored_new[:5]]
        
        # Always keep the global best in the pool
        if best_points is not None:
            current_top.append(best_points.copy())
            # Remove duplicates
            unique_top = []
            for p in current_top:
                is_dup = False
                for u in unique_top:
                    if np.linalg.norm(p - u) < 1e-6:
                        is_dup = True
                        break
                if not is_dup:
                    unique_top.append(p)
            current_top = unique_top[:6]
    
    # Phase 2: Final refinement of the best point found
    if best_points is not None and time.time() < deadline - 10:
        x_best = best_points.flatten()
        # Higher beta for sharper min approximation
        for beta in [500.0, 1000.0]:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    _neg_soft_min_score,
                    x_best,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                res_pts = res.x.reshape(N, 2)
                res_score = _score(res_pts)
                if res_score > best_score:
                    best_score = res_score
                    best_points = res_pts.copy()
                x_best = res.x
            except Exception:
                break
    
    # Fallback
    if best_points is None:
        best_points = _regular_polygon()
        best_score = _score(best_points)
    
    return best_points
