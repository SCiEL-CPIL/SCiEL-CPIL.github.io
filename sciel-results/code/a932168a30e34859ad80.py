"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute triangle indices
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286
TRI_ARR = np.array(TRI_INDICES, dtype=np.int32)  # (286, 3)

def _all_tri_areas(pts):
    """Compute all 286 triangle areas vectorized."""
    i, j, k = TRI_ARR[:,0], TRI_ARR[:,1], TRI_ARR[:,2]
    a = pts[i]
    b = pts[j]
    c = pts[k]
    v1 = b - a
    v2 = c - a
    areas = 0.5 * np.abs(v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0])
    return areas

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 0.0
    areas = _all_tri_areas(pts)
    min_area = np.min(areas)
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _angles_to_pts(angles):
    """Convert sorted angular coordinates on unit circle to 2D points."""
    angles = np.sort(angles) % (2 * np.pi)
    # Ensure strictly increasing with small offset to avoid coincident points
    angles = np.sort(angles)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _pts_to_angles(pts):
    """Convert 2D points to angular coordinates (for points on/near unit circle)."""
    angles = np.arctan2(pts[:,1], pts[:,0])
    angles = np.sort(angles)
    return angles

def _soft_min_obj(angles, temperature):
    """
    Differentiable objective: -softmin(triangle areas) / hull_area.
    Points are on the unit circle, so hull area is computed from the polygon.
    """
    pts = _angles_to_pts(angles)
    # Hull area: since points are on unit circle in sorted angular order,
    # the convex hull IS the polygon formed by them in that order.
    # Area of polygon = 0.5 * sum of cross products of consecutive vertices
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    # Shoelace formula for convex polygon
    hull_area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
    if hull_area < 1e-12:
        return 1e8
    
    areas = _all_tri_areas(pts)
    areas = np.maximum(areas, 1e-15)
    
    # Soft-min via log-sum-exp
    scaled = -areas / temperature
    scaled_max = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - scaled_max))) + scaled_max
    soft_min_area = -log_sum * temperature
    
    return -soft_min_area / hull_area

def _neg_score(x):
    pts = x.reshape(N, 2)
    r = _score(pts)
    return -r

def _anneal_optimize(angles0, time_budget, rng):
    """
    Multi-scale temperature annealing with L-BFGS-B on angular representation.
    """
    start_time = time.time()
    best_angles = angles0.copy()
    best_pts = _angles_to_pts(best_angles)
    best_score = _score(best_pts)
    
    # Temperature schedule: start high, decrease
    temps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for temp in temps:
        if time.time() - start_time > time_budget - 2:
            break
        try:
            def obj(a):
                if time.time() - start_time > time_budget - 1:
                    return 1e8
                return _soft_min_obj(a, temp)
            
            res = minimize(obj, best_angles, method='L-BFGS-B',
                           options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10})
            cand_angles = res.x
            cand_pts = _angles_to_pts(cand_angles)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_angles = cand_angles.copy()
                best_pts = cand_pts.copy()
        except Exception:
            pass
    
    return best_pts, best_score

def _random_angles(rng, n=N):
    """Generate random angular coordinates."""
    return rng.uniform(0, 2*np.pi, n)

def _regular_angles(n=N):
    """Equally spaced angles."""
    return np.linspace(0, 2*np.pi, n, endpoint=False)

def _perturbed_angles(base, rng, scale=0.1):
    """Perturb base angles."""
    return base + rng.normal(0, scale, size=len(base))

def _halton_angles(n, base):
    """Halton sequence mapped to [0, 2*pi)."""
    result = np.zeros(n)
    for i in range(n):
        f = 1.0
        idx = i + 1
        while idx > 0:
            f /= base
            idx = (idx - 1) // base
        result[i] = f
    return result * 2 * np.pi

def construct():
    start_time = time.time()
    total_budget = 145.0  # safety margin for 150s limit
    rng = np.random.RandomState(42)

    def _time_left():
        return total_budget - (time.time() - start_time)

    # Generate diverse initial angle sets
    initial_sets = []
    initial_sets.append(_regular_angles(N))
    initial_sets.append(_halton_angles(N, 2))
    initial_sets.append(_halton_angles(N, 3))
    initial_sets.append(_halton_angles(N, 5))
    for _ in range(8):
        initial_sets.append(_random_angles(rng))
    # Perturbed regular
    for scale in [0.05, 0.1, 0.2]:
        initial_sets.append(_perturbed_angles(_regular_angles(N), rng, scale))

    # Evaluate initial population
    pop_scores = []
    for angles in initial_sets:
        if _time_left() < 3:
            break
        try:
            pts = _angles_to_pts(angles)
            pop_scores.append(_score(pts))
        except Exception:
            pop_scores.append(0.0)
    
    best_idx = int(np.argmax(pop_scores))
    best_score = pop_scores[best_idx]
    best_pts = _angles_to_pts(initial_sets[best_idx])

    # Anneal optimize from top candidates
    # Sort by score descending
    sorted_indices = np.argsort(pop_scores)[::-1]
    top_k = min(5, len(sorted_indices))
    
    for idx in sorted_indices[:top_k]:
        if _time_left() < 10:
            break
        angles = initial_sets[idx].copy()
        time_for_this = min(20.0, _time_left() - 5.0)
        if time_for_this < 5:
            break
        try:
            cand_pts, cand_score = _anneal_optimize(angles, time_for_this, rng)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts.copy()
        except Exception:
            pass
    
    # Final polish on the best with a short Nelder-Mead in Cartesian space
    if _time_left() > 5:
        try:
            res = minimize(_neg_score, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            cand_pts = res.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts.copy()
        except Exception:
            pass

    if best_pts is None:
        best_pts = _angles_to_pts(_regular_angles(N))

    return best_pts
# EVOLVE-BLOCK-END
