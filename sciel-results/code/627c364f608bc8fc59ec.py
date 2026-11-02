"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 13

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples, vectorized."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    # Iterate over pairs (i, j) and vectorize over k > j
    # This is O(n^3) but with low constant factor due to numpy
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area if min_area != float('inf') else 0.0

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_a

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _soft_min_triangle_area(pts, alpha=50.0):
    """
    Smooth approximation of the minimum triangle area.
    Uses -log(sum(exp(-alpha * areas))) / alpha
    This is differentiable and guides optimizers better.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    log_sum = -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            if len(areas) > 0:
                # Shift for numerical stability
                max_a = np.max(areas)
                if max_a > 0:
                    log_sum_partial = np.log(np.sum(np.exp(-alpha * (areas - max_a)))) + (-alpha * max_a)
                    # Actually, we want log(sum(exp(-alpha * a)))
                    # Let M = min(areas) for stability? No, max is better for exp(-x) if x is large?
                    # exp(-alpha * a) is small when a is large.
                    # We want to avoid underflow.
                    # Let's use: log(sum(exp(-alpha * a))) = -alpha * M + log(sum(exp(-alpha * (a - M)))) where M = min(a)?
                    # No, standard log-sum-exp for sum(exp(x_i)) uses max(x_i).
                    # Here x_i = -alpha * a_i. Max(x_i) = -alpha * min(a_i).
                    # So M = min(areas).
                    min_a = np.min(areas)
                    if min_a > 0:
                         shifted = -alpha * (areas - min_a)
                         log_sum_partial = np.log(np.sum(np.exp(shifted))) - alpha * min_a
                         if log_sum_partial > log_sum:
                             log_sum = log_sum_partial
    if log_sum == -np.inf:
        return 0.0
    return -log_sum / alpha

def _hull_area_safe(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 1.0 # Default to 1 if hull fails to avoid division by zero in smooth score

def _smooth_score(pts):
    """Smooth score surrogate."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area_safe(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _soft_min_triangle_area(pts, alpha=50.0) / hull_a

def _neg_smooth_score(params):
    pts = params.reshape(-1, 2)
    return -_smooth_score(pts)

def _generate_initializations():
    """Generate diverse starting points."""
    inits = []
    rng = np.random.RandomState(42)
    
    # 1. Regular Polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Regular Polygon + Center (if N is odd, N-1 on ring, 1 center)
    m = N - 1
    angles_m = np.linspace(0, 2 * np.pi, m, endpoint=False)
    outer = np.column_stack([np.cos(angles_m), np.sin(angles_m)])
    center = np.array([[0.0, 0.0]])
    inits.append(np.vstack([outer, center]))
    
    # 3. Concentric Rings
    # Try splitting N into two rings
    for n1 in range(3, N-2):
        n2 = N - n1
        ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
        r1 = 0.5
        r2 = 1.0
        ring1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
        ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
        inits.append(np.vstack([ring1, ring2]))
        
    # 4. Jittered Regular Polygon
    for jitter in [0.01, 0.05, 0.1]:
        r = 1.0 + jitter * rng.randn(N)
        inits.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
        
    # 5. Random points in unit disk
    for _ in range(5):
        r = rng.rand(N) ** 0.5
        theta = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        
    # 6. Grid-like
    # 4x4 grid minus 3 corners? 
    xs = np.linspace(-0.5, 0.5, 4)
    ys = np.linspace(-0.5, 0.5, 4)
    xx, yy = np.meshgrid(xs, ys)
    pts_grid = np.column_stack([xx.flatten(), yy.flatten()])
    # Select 13 points
    inits.append(pts_grid[:13])
    
    return inits

def _construct_symmetric_config(params, config_type):
    """
    Construct points from low-dimensional parameters.
    params: array of parameters specific to config_type
    config_type: string identifier
    """
    if config_type == "1_12":
        # 1 center, 12 outer
        r_outer, phase = params
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False) + phase
        outer = np.column_stack([r_outer * np.cos(angles), r_outer * np.sin(angles)])
        center = np.array([[0.0, 0.0]])
        return np.vstack([center, outer])
        
    elif config_type == "1_6_6":
        # 1 center, 6 inner, 6 outer
        r_inner, r_outer, phase_inner, phase_outer = params
        ang_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False) + phase_inner
        ang_outer = np.linspace(0, 2 * np.pi, 6, endpoint=False) + phase_outer
        inner = np.column_stack([r_inner * np.cos(ang_inner), r_inner * np.sin(ang_inner)])
        outer = np.column_stack([r_outer * np.cos(ang_outer), r_outer * np.sin(ang_outer)])
        center = np.array([[0.0, 0.0]])
        return np.vstack([center, inner, outer])
        
    elif config_type == "1_5_7":
        # 1 center, 5 inner, 7 outer
        r_inner, r_outer, phase_inner, phase_outer = params
        ang_inner = np.linspace(0, 2 * np.pi, 5, endpoint=False) + phase_inner
        ang_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False) + phase_outer
        inner = np.column_stack([r_inner * np.cos(ang_inner), r_inner * np.sin(ang_inner)])
        outer = np.column_stack([r_outer * np.cos(ang_outer), r_outer * np.sin(ang_outer)])
        center = np.array([[0.0, 0.0]])
        return np.vstack([center, inner, outer])
        
    elif config_type == "1_4_8":
        # 1 center, 4 inner, 8 outer
        r_inner, r_outer, phase_inner, phase_outer = params
        ang_inner = np.linspace(0, 2 * np.pi, 4, endpoint=False) + phase_inner
        ang_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False) + phase_outer
        inner = np.column_stack([r_inner * np.cos(ang_inner), r_inner * np.sin(ang_inner)])
        outer = np.column_stack([r_outer * np.cos(ang_outer), r_outer * np.sin(ang_outer)])
        center = np.array([[0.0, 0.0]])
        return np.vstack([center, inner, outer])
        
    elif config_type == "1_3_9":
        # 1 center, 3 inner, 9 outer
        r_inner, r_outer, phase_inner, phase_outer = params
        ang_inner = np.linspace(0, 2 * np.pi, 3, endpoint=False) + phase_inner
        ang_outer = np.linspace(0, 2 * np.pi, 9, endpoint=False) + phase_outer
        inner = np.column_stack([r_inner * np.cos(ang_inner), r_inner * np.sin(ang_inner)])
        outer = np.column_stack([r_outer * np.cos(ang_outer), r_outer * np.sin(ang_outer)])
        center = np.array([[0.0, 0.0]])
        return np.vstack([center, inner, outer])
        
    return np.zeros((N, 2))

def _neg_symmetric_score(params, config_type):
    pts = _construct_symmetric_config(params, config_type)
    return -_score(pts)

def _optimize_symmetric(config_type, bounds, start_params, deadline):
    """Optimize a symmetric configuration."""
    try:
        # Use Nelder-Mead on the low-dimensional space
        res = minimize(
            _neg_symmetric_score,
            start_params,
            args=(config_type,),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if time.time() > deadline:
            return None, 0.0
            
        best_params = res.x
        pts = _construct_symmetric_config(best_params, config_type)
        score = _score(pts)
        return pts, score
    except Exception:
        return None, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    
    # 1. Evaluate initializations
    inits = _generate_initializations()
    for init_pts in inits:
        if time.time() > deadline - 20.0:
            break
        s = _score(init_pts)
        if s > best_score:
            best_score = s
            best_pts = init_pts.copy()
            
    # 2. Symmetric Search Strategy
    # Define configurations and their parameter bounds
    symmetric_configs = [
        ("1_12", [(0.5, 2.0), (0, 2*np.pi)], [1.0, 0.0]),
        ("1_6_6", [(0.1, 1.0), (0.5, 2.0), (0, 2*np.pi), (0, 2*np.pi)], [0.5, 1.0, 0.0, 0.0]),
        ("1_5_7", [(0.1, 1.0), (0.5, 2.0), (0, 2*np.pi), (0, 2*np.pi)], [0.5, 1.0, 0.0, 0.0]),
        ("1_4_8", [(0.1, 1.0), (0.5, 2.0), (0, 2*np.pi), (0, 2*np.pi)], [0.5, 1.0, 0.0, 0.0]),
        ("1_3_9", [(0.1, 1.0), (0.5, 2.0), (0, 2*np.pi), (0, 2*np.pi)], [0.5, 1.0, 0.0, 0.0]),
    ]
    
    symmetric_candidates = []
    
    for config_type, bounds, start_params in symmetric_configs:
        if time.time() > deadline - 30.0:
            break
            
        pts, score = _optimize_symmetric(config_type, bounds, start_params, deadline)
        if pts is not None:
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
            symmetric_candidates.append(pts)
            
    # 3. Optimization Loop on Best Candidates
    # We will try multiple restarts from the best few initializations, symmetric results, and the current best
    candidates = []
    # Add top 3 initializations
    scored_inits = [(_score(p), p) for p in inits]
    scored_inits.sort(key=lambda x: -x[0])
    for s, p in scored_inits[:3]:
        candidates.append(p)
        
    # Add symmetric candidates
    for p in symmetric_candidates:
        candidates.append(p)
        
    candidates.append(best_pts)
    
    # Remove duplicates roughly
    unique_candidates = []
    for c in candidates:
        if not any(np.allclose(c, u, atol=1e-5) for u in unique_candidates):
            unique_candidates.append(c)
            
    for init_pts in unique_candidates:
        if time.time() > deadline - 15.0:
            break
            
        current_pts = init_pts.copy()
        current_score = _score(current_pts)
        
        # Step A: Smooth Refinement
        # Use L-BFGS-B or Powell on smooth score
        # Powell is often more robust for non-convex problems without gradients
        try:
            res_smooth = minimize(
                _neg_smooth_score,
                current_pts.flatten(),
                method='Powell',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            smooth_pts = res_smooth.x.reshape(-1, 2)
            smooth_score = _score(smooth_pts) # Check true score
            if smooth_score > current_score:
                current_pts = smooth_pts
                current_score = smooth_score
        except Exception:
            pass
        
        # Step B: Exact Refinement
        # Use Nelder-Mead on true score
        try:
            res_exact = minimize(
                _neg_score,
                current_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            exact_pts = res_exact.x.reshape(-1, 2)
            exact_score = _score(exact_pts)
            if exact_score > current_score:
                current_pts = exact_pts
                current_score = exact_score
        except Exception:
            pass
            
        # Step C: Another round of smooth + exact if time permits
        if time.time() < deadline - 10.0:
            try:
                res_smooth2 = minimize(
                    _neg_smooth_score,
                    current_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-8}
                )
                smooth_pts2 = res_smooth2.x.reshape(-1, 2)
                smooth_score2 = _score(smooth_pts2)
                if smooth_score2 > current_score:
                    current_pts = smooth_pts2
                    current_score = smooth_score2
            except Exception:
                pass
            
            try:
                res_exact2 = minimize(
                    _neg_score,
                    current_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                exact_pts2 = res_exact2.x.reshape(-1, 2)
                exact_score2 = _score(exact_pts2)
                if exact_score2 > current_score:
                    current_pts = exact_pts2
                    current_score = exact_score2
            except Exception:
                pass
        
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()

    # 4. Global Search with Differential Evolution (if time allows)
    # DE is stochastic and can find better basins.
    if time.time() < deadline - 30.0:
        # Define bounds: points should be within a reasonable range, e.g., [-2, 2]
        bounds = [(-2, 2)] * (2 * N)
        
        # We want to maximize score, so minimize -score
        # DE might be slow, so limit iterations
        try:
            # Use a subset of the best points as initial population? DE generates its own.
            # But we can seed it with the current best if we use 'init' param? 
            # Actually, standard DE is enough if we give it enough time.
            # Let's try a quick DE run.
            res_de = differential_evolution(
                _neg_score,
                bounds,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                seed=42,
                polish=True # This runs L-BFGS-B at the end
            )
            de_pts = res_de.x.reshape(-1, 2)
            de_score = _score(de_pts)
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts.copy()
        except Exception:
            pass
            
    # 5. Final Polish on Best
    if time.time() < deadline - 5.0:
        try:
            res_final = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            final_pts = res_final.x.reshape(-1, 2)
            final_score = _score(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts.copy()
        except Exception:
            pass

    return best_pts

# EVOLVE-BLOCK-END
