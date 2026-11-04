"""Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute triangle indices for vectorized area computation
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]
_I, _J, _K = _TRIPLETS.T

def _triangle_areas_vectorized(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    a = pts[_I]
    b = pts[_J]
    c = pts[_K]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    return 0.5 * np.abs(cross)

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 0.0
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas) / hull_area

def _soft_min_ratio(pts, temperature):
    """
    Smooth surrogate: soft-min of triangle areas / hull area.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return -1e9
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return -1e9
    except Exception:
        return -1e9
    
    areas = _triangle_areas_vectorized(pts)
    areas = np.maximum(areas, 1e-15)
    
    scaled = -areas / temperature
    scaled_max = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - scaled_max))) + scaled_max
    soft_min_area = -log_sum * temperature
    
    return soft_min_area / hull_area

def _halton_sequence(n, base):
    """Generate n Halton sequence points in [0,1)."""
    result = []
    f = 1.0
    for i in range(1, n + 1):
        x = 0.0
        i_val = i
        while i_val > 0:
            i_val, digit = divmod(i_val, base)
            x += digit * f
            f /= base
        result.append(x)
    return np.array(result)

def _generate_halton_points(n, base1=2, base2=3):
    """Generate n points using 2D Halton sequence in unit square, mapped to disk."""
    x = _halton_sequence(n, base1)
    y = _halton_sequence(n, base2)
    r = np.sqrt(x)
    theta = 2 * np.pi * y
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return pts

def _regular_polygon(n, radius=1.0, center=(0,0)):
    """Generate regular polygon vertices."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius*np.cos(angles), 
                            center[1] + radius*np.sin(angles)])

def _concentric_rings(n, n_rings=3):
    """Generate points on concentric rings for diverse initialization."""
    pts = []
    pts.append([0.0, 0.0])
    remaining = n - 1
    for ring in range(n_rings):
        if remaining <= 0:
            break
        n_on_ring = min(remaining, max(3, (n-1)//n_rings))
        if ring == n_rings - 1:
            n_on_ring = remaining
        radius = (ring + 1) / n_rings
        angles = np.linspace(0, 2*np.pi, n_on_ring, endpoint=False) + ring * np.pi/6
        ring_pts = np.column_stack([radius*np.cos(angles), radius*np.sin(angles)])
        pts.append(ring_pts)
        remaining -= n_on_ring
    if remaining > 0:
        angles = np.linspace(0, 2*np.pi, remaining, endpoint=False)
        ring_pts = np.column_stack([1.0*np.cos(angles), 1.0*np.sin(angles)])
        pts.append(ring_pts)
    result = np.vstack(pts)
    return result[:n]

def _structured_grid(n):
    """Generate points on a structured grid pattern."""
    xs = np.linspace(-0.9, 0.9, 4)
    ys = np.linspace(-0.9, 0.9, 4)
    grid_pts = np.array([[x, y] for x in xs for y in ys])
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(grid_pts))[:n]
    return grid_pts[idx]

def _optimize_from_start(initial_pts, time_budget, rng):
    """
    Optimize from a starting point using multi-stage approach:
    1. L-BFGS with smooth surrogate (multiple temperatures)
    2. Nelder-Mead on exact objective
    """
    start_time = time.time()
    best_pts = initial_pts.copy()
    best_score = _score(initial_pts)
    
    # Stage 1: L-BFGS with decreasing temperature
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002]
    x0 = best_pts.flatten().copy()
    
    for temp in temperatures:
        if time.time() - start_time > time_budget - 3:
            break
        
        def obj(x):
            pts = x.reshape(N, 2)
            ratio = _soft_min_ratio(pts, temp)
            if ratio < -1e8:
                return 1e8
            return -ratio
        
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8,
                         'maxls': 40}
            )
            cand = result.x
            cand_score = _score(cand.reshape(N, 2))
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand.reshape(N, 2).copy()
            x0 = cand
        except Exception:
            pass
    
    # Stage 2: Nelder-Mead polish on exact objective
    if time.time() - start_time < time_budget - 1:
        def exact_obj(x):
            pts = x.reshape(N, 2)
            r = _score(pts)
            return -r
        
        try:
            result = minimize(
                exact_obj, best_pts.flatten(), method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11,
                         'initial_simplex': 1e-3}
            )
            cand = result.x
            cand_score = _score(cand.reshape(N, 2))
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand.reshape(N, 2).copy()
        except Exception:
            pass
    
    return best_pts, best_score

def construct():
    """Main entry point: find optimal 13-point configuration."""
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s limit
    
    rng = np.random.RandomState(12345)
    
    best_points = None
    best_ratio = -1e9
    
    # Generate diverse starting points using Halton sequences and structured patterns
    starts = []
    
    # 1. Regular 13-gon
    starts.append(_regular_polygon(N, radius=1.0))
    
    # 2-4: Halton sequences with different bases
    starts.append(_generate_halton_points(N, base1=2, base2=3))
    starts.append(_generate_halton_points(N, base1=3, base2=5))
    starts.append(_generate_halton_points(N, base1=5, base2=7))
    
    # 5. Concentric rings
    starts.append(_concentric_rings(N, n_rings=3))
    
    # 6. Structured grid
    starts.append(_structured_grid(N))
    
    # 7-9: Jittered Halton
    for seed_offset in [1, 2, 3]:
        halton_pts = _generate_halton_points(N, base1=2, base2=3)
        jitter = rng.normal(0, 0.08, (N, 2))
        starts.append(halton_pts + jitter)
    
    # Evaluate initial candidates
    for cand in starts:
        if time.time() - start_time > total_budget:
            break
        try:
            s = _score(cand)
            if s > best_ratio:
                best_ratio = s
                best_points = cand.copy()
        except Exception:
            pass
    
    # Sort starts by initial score (best first) to allocate more time to promising starts
    scored_starts = []
    for start in starts:
        try:
            s = _score(start)
            scored_starts.append((s, start))
        except Exception:
            scored_starts.append((0.0, start))
    scored_starts.sort(key=lambda x: -x[0])
    
    # Optimize from each starting point
    n_starts = len(starts)
    # Allocate time proportional to initial quality, with minimum
    base_time = (total_budget - 5.0) / n_starts
    
    for i, (init_score, start) in enumerate(scored_starts):
        if time.time() - start_time > total_budget - 2:
            break
        
        remaining_time = total_budget - (time.time() - start_time)
        # Give more time to better initial points
        quality_factor = 1.0 + 0.5 * (init_score / max(best_ratio, 1e-9))
        alloc_time = min(base_time * quality_factor, remaining_time - 1.0)
        if alloc_time < 2.0:
            continue
        
        try:
            opt_pts, opt_score = _optimize_from_start(start, alloc_time, rng)
            if opt_score > best_ratio:
                best_ratio = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Final refinement on the best found
    if best_points is not None and time.time() - start_time < total_budget - 3:
        final_time = total_budget - (time.time() - start_time) - 1.0
        if final_time > 8.0:
            try:
                final_pts, final_score = _optimize_from_start(best_points, final_time, rng)
                if final_score > best_ratio:
                    best_ratio = final_score
                    best_points = final_pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: regular polygon
        best_points = _regular_polygon(N, radius=1.0)
    
    return best_points
# EVOLVE-BLOCK-END
