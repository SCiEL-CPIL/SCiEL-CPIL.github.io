import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin

def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

TRIPLE_INDICES = _get_triple_indices(N)
I_IDX = TRIPLE_INDICES[:, 0]
J_IDX = TRIPLE_INDICES[:, 1]
K_IDX = TRIPLE_INDICES[:, 2]
NUM_TRIPLES = len(TRIPLE_INDICES)

def _score(points):
    """Compute min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    p_i = points[I_IDX]
    p_j = points[J_IDX]
    p_k = points[K_IDX]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _neg_score(flat_points):
    points = flat_points.reshape((N, 2))
    return -_score(points)

def _gradient_neg_score(flat_points):
    """Compute gradient of negative score (approximate via finite differences)."""
    points = flat_points.reshape((N, 2))
    score = _score(points)
    grad = np.zeros_like(flat_points)
    eps = 1e-6
    
    for idx in range(len(flat_points)):
        perturbed = flat_points.copy()
        perturbed[idx] += eps
        new_score = _score(perturbed.reshape((N, 2)))
        grad[idx] = (new_score - score) / eps
    
    return -grad  # gradient of negative score

def _make_regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _make_concentric_rings(rings):
    """rings: list of (count, radius, angle_offset)"""
    pts = []
    for count, radius, offset in rings:
        angles = np.linspace(offset, 2*np.pi + offset, count, endpoint=False)
        pts.append(np.column_stack([radius * np.cos(angles), radius * np.sin(angles)]))
    return np.vstack(pts)

def _random_points(n, scale=1.0):
    return np.random.randn(n, 2) * scale

def _optimize_from_start(start_points, max_iter=300):
    """Optimize from a starting configuration using L-BFGS-B with finite diff gradient."""
    flat = start_points.flatten()
    
    # First try Nelder-Mead for robustness
    try:
        res = minimize(
            _neg_score,
            flat,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'maxfev': max_iter * 2, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        opt_points = res.x.reshape((N, 2))
        opt_score = _score(opt_points)
    except Exception:
        opt_points = start_points.copy()
        opt_score = _score(start_points)
    
    # Then try L-BFGS-B with gradient for further refinement
    if time.time() < DEADLINE - 10:
        try:
            res2 = minimize(
                _neg_score,
                opt_points.flatten(),
                method='L-BFGS-B',
                jac=_gradient_neg_score,
                options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-12}
            )
            opt_points2 = res2.x.reshape((N, 2))
            opt_score2 = _score(opt_points2)
            if opt_score2 > opt_score:
                opt_points = opt_points2
                opt_score = opt_score2
        except Exception:
            pass
    
    return opt_points, opt_score

def _local_refine(points, steps=50):
    """Local refinement: nudge points to increase the minimum triangle area."""
    current = points.copy()
    current_score = _score(current)
    
    for step in range(steps):
        if time.time() > DEADLINE - 5:
            break
        
        # Find the triple with minimum area
        p_i = current[I_IDX]
        p_j = current[J_IDX]
        p_k = current[K_IDX]
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_idx = np.argmin(areas)
        i, j, k = TRIPLE_INDICES[min_idx]
        
        # Try to increase the area of this triangle by moving points
        # Move points i, j, k slightly apart
        pi = current[i].copy()
        pj = current[j].copy()
        pk = current[k].copy()
        
        # Compute centroid
        centroid = (pi + pj + pk) / 3.0
        
        # Move each point away from centroid slightly
        for idx, p in [(i, pi), (j, pj), (k, pk)]:
            direction = p - centroid
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            else:
                direction = np.array([1.0, 0.0])
            
            # Try small perturbation
            for step_size in [0.01, 0.005, 0.001]:
                new_p = current.copy()
                new_p[idx] = p + direction * step_size
                new_score = _score(new_p)
                if new_score > current_score:
                    current = new_p
                    current_score = new_score
                    break
    
    return current, current_score

def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0
    
    np.random.seed(42)
    
    candidates = []
    
    # 1. Regular 14-gon
    candidates.append(_make_regular_polygon(N))
    
    # 2. Two concentric heptagons
    candidates.append(_make_concentric_rings([
        (7, 0.5, 0),
        (7, 1.0, np.pi/7)
    ]))
    
    # 3. Three rings: 5, 5, 4
    candidates.append(_make_concentric_rings([
        (5, 0.3, 0),
        (5, 0.7, 2*np.pi/5),
        (4, 1.0, np.pi/4)
    ]))
    
    # 4. Two rings: 7, 7 with different offsets
    candidates.append(_make_concentric_rings([
        (7, 0.6, 0),
        (7, 1.0, np.pi/14)
    ]))
    
    # 5. Random starts
    for _ in range(8):
        candidates.append(_random_points(N, scale=0.8))
    
    # 6. Slightly perturbed regular polygon
    base = _make_regular_polygon(N)
    for _ in range(4):
        perturbed = base + np.random.randn(N, 2) * 0.1
        candidates.append(perturbed)
    
    best_score = -1
    best_points = None
    
    # Evaluate all candidates
    for init in candidates:
        if time.time() > DEADLINE - 20:
            break
        s = _score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()
    
    # Optimize top candidates
    scored_candidates = []
    for init in candidates:
        s = _score(init)
        scored_candidates.append((s, init))
    
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Optimize the top 6 candidates
    for rank, (s, init) in enumerate(scored_candidates[:6]):
        if time.time() > DEADLINE - 15:
            break
        
        opt_points, opt_score = _optimize_from_start(init, max_iter=200)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
    
    # Local refinement on the best
    if time.time() < DEADLINE - 5 and best_points is not None:
        refined_points, refined_score = _local_refine(best_points, steps=30)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()
    
    # Final quick polish with Nelder-Mead
    if time.time() < DEADLINE - 3 and best_points is not None:
        try:
            res = minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 100, 'maxfev': 200, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            final_points = res.x.reshape((N, 2))
            final_score = _score(final_points)
            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
        except Exception:
            pass
    
    if best_points is None:
        best_points = _make_regular_polygon(N)
    
    return best_points
