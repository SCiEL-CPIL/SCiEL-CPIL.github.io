"""Combined program: angular/radial parameterization + vectorized soft-min optimization."""
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
    # soft_min(x) = -(1/beta) * log(sum(exp(-beta * x_i)))
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    return -soft_min_area / hull_area

def _exact_objective(x):
    """Exact negative score for Nelder-Mead refinement."""
    pts = x.reshape(-1, 2)
    return -_score(pts)

def _initial_points(seed=0):
    """Generate initial points: regular polygon with slight perturbation."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    pts += rng.normal(0, 0.05, size=pts.shape)
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

def _halton_points(n, base1=2, base2=3):
    """Generate n points using 2D Halton sequence, scaled to unit circle."""
    x = _halton_sequence(n, base1)
    y = _halton_sequence(n, base2)
    pts = np.column_stack([x, y])
    # Normalize to unit disk
    radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    radii[radii < 1e-10] = 1.0
    pts = pts / radii[:, np.newaxis]
    return pts

def _center_ring(n_outer, n_inner, r_outer=1.0, r_inner=0.5):
    """Points on two concentric rings."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    return np.vstack([outer, inner])

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    best_score = 0.0
    best_pts = _initial_points(0)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Regular 13-gon
    candidates.append(_initial_points(0))
    
    # 2. Concentric rings (various splits)
    for n_outer in [7, 8, 9, 10]:
        n_inner = N - n_outer
        if n_inner >= 3:
            for r_inner in [0.4, 0.5, 0.6]:
                candidates.append(_center_ring(n_outer, n_inner, 1.0, r_inner))
    
    # 3. Halton sequence points
    candidates.append(_halton_points(N))
    
    # 4. Random points on/near circle
    rng = np.random.default_rng(42)
    for i in range(10):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = rng.uniform(0.7, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # 5. Perturbed regular polygon
    for i in range(10):
        noise = rng.normal(0, 0.05, (N, 2))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        base = np.column_stack([np.cos(angles), np.sin(angles)])
        candidates.append(base + noise)
    
    # Evaluate all candidates with exact score
    for pts in candidates:
        if time.time() > deadline:
            break
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Sort candidates by score, optimize top ones
    scored_candidates = []
    for pts in candidates:
        s = _score(pts)
        scored_candidates.append((s, pts))
    scored_candidates.sort(key=lambda x: -x[0])
    
    top_candidates = [pts for _, pts in scored_candidates[:min(8, len(scored_candidates))]]
    
    # Progressive beta: start smooth, get sharper
    betas = [20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline - 20:
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
                
                if res_score > best_score:
                    best_score = res_score
                    best_pts = res_pts.copy()
                    
            except Exception:
                pass
        
        # Update top candidates with current best
        if best_pts is not None:
            top_candidates = [best_pts.copy()]
    
    # Final refinement: use exact score with Nelder-Mead on the best point
    if best_pts is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                _exact_objective,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 2000}
            )
            res_pts = res.x.reshape(N, 2)
            res_score = _score(res_pts)
            
            if res_score > best_score:
                best_score = res_score
                best_pts = res_pts.copy()
                
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
