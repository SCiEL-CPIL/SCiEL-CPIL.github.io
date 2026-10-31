import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute all triple indices for vectorized triangle area computation
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

# Fixed normal angles for support-line representation (evenly spaced)
_NORMAL_ANGLES = np.linspace(0, 2 * np.pi, N, endpoint=False)
_NORMAL_COS = np.cos(_NORMAL_ANGLES)
_NORMAL_SIN = np.sin(_NORMAL_ANGLES)
# Tangent vectors (perpendicular to normals, rotated 90 degrees CCW)
_TAN_COS = np.cos(_NORMAL_ANGLES + np.pi / 2)
_TAN_SIN = np.sin(_NORMAL_ANGLES + np.pi / 2)


def _triangle_areas_vectorized(points):
    """Compute areas of all C(N,3) triangles in vectorized form."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _polygon_from_offsets(offsets):
    """
    Compute the N vertices of a convex polygon defined by N support lines
    with fixed normal directions. Each offset d_i defines the line:
        n_i · x = d_i
    where n_i is the i-th normal. The vertex i is the intersection of
    lines i and (i+1) mod N.
    """
    # For each adjacent pair of lines, solve:
    #   n_i · v = d_i
    #   n_{i+1} · v = d_{i+1}
    # Using 2x2 linear system.
    vertices = np.zeros((N, 2))
    for i in range(N):
        j = (i + 1) % N
        # Build 2x2 system: [n_i; n_j] @ v = [d_i; d_j]
        A = np.array([[_NORMAL_COS[i], _NORMAL_SIN[i]],
                      [_NORMAL_COS[j], _NORMAL_SIN[j]]])
        b = np.array([offsets[i], offsets[j]])
        try:
            vertices[i] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback: use midpoint of normals scaled
            vertices[i] = offsets[i] * np.array([_NORMAL_COS[i], _NORMAL_SIN[i]])
    return vertices


def _points_from_offsets(offsets):
    """Convert offset parameters to point coordinates (same as polygon_from_offsets)."""
    return _polygon_from_offsets(offsets)


def _offsets_from_points(points):
    """
    Extract support-line offsets from a set of points.
    For each normal direction, compute the max projection.
    This gives the tightest enclosing polygon with those normals.
    """
    projections = points @ np.column_stack([_NORMAL_COS, _NORMAL_SIN]).T
    return np.max(projections, axis=0)


def _neg_score_offsets(offsets):
    """Negative score when parameters are support-line offsets."""
    points = _polygon_from_offsets(offsets)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _neg_soft_score_offsets(offsets, temperature):
    """Negative soft score when parameters are support-line offsets."""
    points = _polygon_from_offsets(offsets)
    s = _soft_score(points, temperature)
    if s <= 0:
        return 1e10
    return -s


def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(points):
    """Negative score for minimization."""
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _soft_min_value(values, temperature):
    """Smooth approximation of min using log-sum-exp."""
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(points, temperature):
    """Score using soft-min instead of hard min."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.any(areas < 1e-15):
        return 0.0
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(points, temperature):
    """Negative soft score for minimization."""
    s = _soft_score(points, temperature)
    if s <= 0:
        return 1e10
    return -s


# ---------- Seed generators ----------

def _seed_regular():
    """Regular 13-gon."""
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_regular_offsets():
    """Regular 13-gon as offsets (all equal)."""
    # For a regular N-gon inscribed in unit circle, the support function
    # value for normal at angle alpha is cos(pi/N) (apothem of unit circumradius polygon)
    # Actually for regular N-gon with circumradius R, apothem = R*cos(pi/N)
    # Normal directions are at angles 0, 2pi/N, ... which align with vertex normals
    # The offset = R * cos(pi/N) for circumradius R
    return np.full(N, np.cos(np.pi / N))


def _seed_perturbed_offsets(seed=42):
    """Perturbed regular polygon offsets."""
    rng = np.random.RandomState(seed)
    base = np.cos(np.pi / N)
    offsets = base + 0.05 * rng.randn(N)
    return np.clip(offsets, 0.3, 2.0)


def _seed_two_ring_offsets(n_out, r_inner, angle_offset):
    """
    Create a two-ring-like configuration in offset space.
    We place N points on two concentric circles, then compute their
    support-line offsets.
    """
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    pts = np.vstack([outer, inner])
    return _offsets_from_points(pts)


def _seed_three_ring_offsets(n1, n2, n3, r2, r3, off2, off3):
    """Three-ring configuration converted to offsets."""
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return _offsets_from_points(pts)


def _seed_quasi_regular_offsets(seed=42):
    """Quasi-regular as offsets."""
    rng = np.random.RandomState(seed)
    base = np.cos(np.pi / N)
    offsets = base + 0.03 * rng.randn(N)
    return np.clip(offsets, 0.5, 1.5)


def _seed_centered_rings_offsets():
    """Centered rings converted to offsets."""
    pts = np.zeros((N, 2))
    for i in range(6):
        a = 2 * np.pi * i / 6
        pts[1 + i] = [0.5 * np.cos(a), 0.5 * np.sin(a)]
    for i in range(6):
        a = 2 * np.pi * i / 6
        pts[7 + i] = [1.0 * np.cos(a), 1.0 * np.sin(a)]
    return _offsets_from_points(pts)


# ---------- Optimization routines ----------

def _annealed_soft_minimize_offsets(offsets, deadline, max_temp=0.10, min_temp=1e-5, n_steps=6):
    """
    Anneal the soft-min temperature and optimize offsets at each step.
    Uses L-BFGS-B then Powell at each temperature level.
    """
    best = offsets.copy()
    best_pts = _polygon_from_offsets(best)
    best_s = _score(best_pts)
    
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    
    for temp in temps:
        if time.time() >= deadline - 8:
            break
        
        # L-BFGS-B on soft objective in offset space
        if time.time() < deadline - 6:
            try:
                def obj(x):
                    return _neg_soft_score_offsets(x, temp)
                
                res = minimize(
                    obj, best.copy(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12},
                )
                cand = res.x
                cand_pts = _polygon_from_offsets(cand)
                s = _score(cand_pts)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
                    best_pts = cand_pts
            except Exception:
                pass
        
        # Powell on soft objective in offset space
        if time.time() < deadline - 4:
            try:
                def obj2(x):
                    return _neg_soft_score_offsets(x, temp)
                
                res = minimize(
                    obj2, best.copy(),
                    method='Powell',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10},
                )
                cand = res.x
                cand_pts = _polygon_from_offsets(cand)
                s = _score(cand_pts)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
                    best_pts = cand_pts
            except Exception:
                pass
    
    return best_pts, best_s


def _polish_true_objective(points, deadline):
    """
    Polish using the true (non-smooth) objective.
    First convert to offsets, optimize in offset space, then do Cartesian polish.
    """
    best = points.copy()
    best_s = _score(best)
    
    # Convert to offsets and optimize in offset space
    if time.time() < deadline - 10:
        try:
            offsets = _offsets_from_points(best)
            
            # Powell in offset space
            res = minimize(
                _neg_score_offsets,
                offsets.copy(),
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-11, 'fatol': 1e-11},
            )
            cand_off = res.x
            cand_pts = _polygon_from_offsets(cand_off)
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best = cand_pts.copy()
        except Exception:
            pass
    
    # Nelder-Mead in Cartesian for fine adjustments
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    # Coordinate descent on hard objective for precise single-point adjustments
    if time.time() < deadline - 2:
        try:
            best, best_s = _coordinate_descent_polish(best, best_s, deadline)
        except Exception:
            pass
    
    return best, best_s


def _coordinate_descent_polish(points, best_s, deadline):
    """
    Systematic coordinate descent on the hard objective.
    For each point, try small displacements in 8 directions with adaptive step sizes.
    This makes fine single-coordinate adjustments that Nelder-Mead may miss.
    """
    best = points.copy()
    s = best_s
    rng = np.random.RandomState(7)
    
    directions = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [0.7071, 0.7071], [-0.7071, 0.7071], [0.7071, -0.7071], [-0.7071, -0.7071]
    ])
    
    # Multiple passes with decreasing step sizes
    for pass_num in range(3):
        if time.time() >= deadline - 2:
            break
        
        base_sigma = 0.005 * (0.5 ** pass_num)
        
        for idx in range(N):
            if time.time() >= deadline - 2:
                break
            
            for d in directions:
                for scale in [1.0, 0.5, 0.25]:
                    step = base_sigma * scale
                    cand = best.copy()
                    cand[idx] += step * d
                    
                    s_new = _score(cand)
                    if s_new > s + 1e-14:
                        s = s_new
                        best = cand.copy()
    
    return best, s


def _local_search(points, deadline, n_iter=300, sigma_start=0.015, sigma_end=0.0005):
    """
    Coordinate-wise local search with adaptive step size.
    Tries moving each point in random directions, keeping improvements.
    """
    best = points.copy()
    best_s = _score(best)
    rng = np.random.RandomState(123)
    
    for it in range(n_iter):
        if time.time() >= deadline - 4:
            break
        
        # Adaptive sigma
        progress = it / max(n_iter - 1, 1)
        sigma = sigma_start * (sigma_end / sigma_start) ** progress
        
        # Pick a random point and a random direction
        idx = rng.randint(N)
        direction = rng.randn(2)
        direction /= (np.linalg.norm(direction) + 1e-15)
        
        # Try a few step sizes
        for scale in [1.0, 0.5, 0.2, 2.0]:
            step = sigma * scale
            cand = best.copy()
            cand[idx] += step * direction
            
            s = _score(cand)
            if s > best_s + 1e-15:
                best_s = s
                best = cand.copy()
                break
    
    return best, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Generate diverse initializations in offset space
    offset_inits = []
    offset_inits.append(_seed_regular_offsets())
    offset_inits.append(_seed_perturbed_offsets(42))
    offset_inits.append(_seed_perturbed_offsets(123))
    offset_inits.append(_seed_perturbed_offsets(777))
    offset_inits.append(_seed_quasi_regular_offsets(42))
    offset_inits.append(_seed_quasi_regular_offsets(999))
    offset_inits.append(_seed_two_ring_offsets(7, 0.5, 0.0))
    offset_inits.append(_seed_two_ring_offsets(6, 0.4, np.pi / 6))
    offset_inits.append(_seed_two_ring_offsets(8, 0.6, np.pi / 4))
    offset_inits.append(_seed_three_ring_offsets(5, 4, 4, 0.5, 0.8, 0.0, np.pi / 5))
    offset_inits.append(_seed_three_ring_offsets(6, 3, 4, 0.4, 0.7, np.pi / 6, np.pi / 3))
    offset_inits.append(_seed_centered_rings_offsets())
    
    # Also add Cartesian seeds converted to offsets
    cart_seeds = [_seed_regular()]
    rng = np.random.RandomState(42)
    for _ in range(3):
        cart_seeds.append(rng.random((N, 2)))
    for cs in cart_seeds:
        offset_inits.append(_offsets_from_points(cs))
    
    best_points = None
    best_score = -1
    
    # Phase 1: Annealed soft-min optimization in offset space on each seed
    n_inits = len(offset_inits)
    for i, init_off in enumerate(offset_inits):
        if time.time() >= deadline - 20:
            break
        
        remaining_seeds = n_inits - i
        time_per_seed = min(10.0, max(3.0, (deadline - 20 - time.time()) / max(remaining_seeds, 1)))
        seed_deadline = min(deadline - 20, time.time() + time_per_seed)
        
        try:
            pts, s = _annealed_soft_minimize_offsets(init_off, seed_deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 2: Polish the best result with true objective
    if best_points is not None:
        if time.time() < deadline - 12:
            try:
                pts, s = _polish_true_objective(best_points, deadline)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
            except Exception:
                pass
    
    # Phase 3: Local search refinement
    if best_points is not None:
        if time.time() < deadline - 6:
            try:
                pts, s = _local_search(best_points, deadline)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish
    if best_points is not None and time.time() < deadline - 3:
        try:
            # Try offset-space Nelder-Mead
            off = _offsets_from_points(best_points)
            res = minimize(
                _neg_score_offsets,
                off.copy(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True},
            )
            pts = _polygon_from_offsets(res.x)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 5: Final coordinate descent polish
    if best_points is not None and time.time() < deadline - 1:
        try:
            pts, s = _coordinate_descent_polish(best_points, best_score, deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = _seed_regular()
    
    return best_points
