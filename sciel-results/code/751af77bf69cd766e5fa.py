import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13

# Pre-compute triangle indices for N=13
_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_I_IDX = _TRI_INDICES[:, 0]
_J_IDX = _TRI_INDICES[:, 1]
_K_IDX = _TRI_INDICES[:, 2]
_NUM_TRI = len(_TRI_INDICES)

# Fixed normal angles for the 13 half-planes
_THETA = 2 * np.pi * np.arange(N) / N
_NORM_X = np.cos(_THETA)
_NORM_Y = np.sin(_THETA)

def _offsets_to_points(offsets):
    """
    Convert 13 support offsets to 13 polygon vertices.
    Each side k has equation: n_k . x = d_k, where n_k = (cos(theta_k), sin(theta_k))
    Vertex k is intersection of sides k-1 and k.
    """
    d = np.asarray(offsets)
    points = np.zeros((N, 2))
    for k in range(N):
        k_prev = (k - 1) % N
        # Solve: n_prev . x = d_prev, n_k . x = d_k
        # [nx_prev, ny_prev] [x]   [d_prev]
        # [nx_k,     ny_k  ] [y] = [d_k   ]
        a11, a12 = _NORM_X[k_prev], _NORM_Y[k_prev]
        a21, a22 = _NORM_X[k], _NORM_Y[k]
        det = a11 * a22 - a12 * a21
        if abs(det) < 1e-15:
            points[k] = [0.0, 0.0]
        else:
            x = (d[k_prev] * a22 - d[k] * a12) / det
            y = (a11 * d[k] - a21 * d[k_prev]) / det
            points[k] = [x, y]
    return points

def _get_triangle_areas(points):
    """Vectorized calculation of areas for all triples."""
    pi = points[_I_IDX]
    pj = points[_J_IDX]
    pk = points[_K_IDX]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the raw score: min_triangle_area / hull_area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _score_from_offsets(offsets):
    """Score computed from offset parameterization."""
    points = _offsets_to_points(offsets)
    return _score(points)

def _neg_score_from_offsets(offsets):
    """Negative score for minimization."""
    s = _score_from_offsets(offsets)
    return -s

def _softmin_score_from_offsets(offsets, alpha):
    """Softmin-based smooth score from offsets."""
    points = _offsets_to_points(offsets)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    a_min = np.min(areas)
    exp_terms = np.exp(-alpha * (areas - a_min))
    log_sum = np.log(np.sum(exp_terms))
    softmin_val = a_min - np.log(np.sum(exp_terms)) / alpha
    return softmin_val / hull_area

def _neg_softmin_from_offsets(offsets, alpha):
    """Negative softmin score for minimization."""
    return -_softmin_score_from_offsets(offsets, alpha)

def _regular_offsets():
    """Uniform offsets giving a regular 13-gon."""
    # For a regular polygon with circumradius R, the apothem (inradius) is R*cos(pi/N)
    # We want unit-area polygon. Area of regular N-gon with inradius r: A = N * r^2 * tan(pi/N)
    # So r = sqrt(A / (N * tan(pi/N)))
    r = np.sqrt(1.0 / (N * np.tan(np.pi / N)))
    return np.full(N, r)

def _optimize_offsets(offsets_init, deadline):
    """Multi-stage optimization in offset space."""
    best_offsets = offsets_init.copy()
    best_score = _score_from_offsets(best_offsets)
    
    # Stage 1: Softmin with decreasing alpha
    alphas = [50, 25, 10, 5, 2, 1]
    for alpha in alphas:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                _neg_softmin_from_offsets,
                best_offsets,
                args=(alpha,),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            cand_offsets = res.x
            s = _score_from_offsets(cand_offsets)
            if s > best_score:
                best_score = s
                best_offsets = cand_offsets
        except Exception:
            pass
    
    # Stage 2: Nelder-Mead on hard score
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _neg_score_from_offsets,
                best_offsets,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            cand_offsets = res.x
            s = _score_from_offsets(cand_offsets)
            if s > best_score:
                best_score = s
                best_offsets = cand_offsets
        except Exception:
            pass
    
    return best_offsets, best_score

def _polish_offsets(offsets, deadline):
    """Coordinate descent polish in offset space."""
    best_offsets = offsets.copy()
    best_score = _score_from_offsets(best_offsets)
    
    for outer in range(30):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            orig = best_offsets[i]
            best_val = orig
            best_s = best_score
            for delta in [-0.05, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.05]:
                cand = best_offsets.copy()
                cand[i] = orig + delta
                s = _score_from_offsets(cand)
                if s > best_s + 1e-12:
                    best_s = s
                    best_val = orig + delta
            if best_val != orig:
                best_offsets[i] = best_val
                improved = True
        if not improved:
            break
        cur_s = _score_from_offsets(best_offsets)
        if cur_s > best_score:
            best_score = cur_s
    
    return best_offsets, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)
    
    # Generate initial offset configurations
    initial_offsets_list = []
    
    # Regular polygon (all offsets equal)
    initial_offsets_list.append(_regular_offsets())
    
    # Perturbed regular
    base = _regular_offsets()
    for amp in [0.05, 0.1, 0.2, 0.3]:
        for freq in [2, 3, 4, 5]:
            perturbation = amp * np.sin(freq * _THETA)
            initial_offsets_list.append(base + perturbation)
    
    # Random offsets near regular
    for _ in range(10):
        noise = rng.normal(0, 0.1, N)
        initial_offsets_list.append(base + noise)
    
    # More diverse random
    for _ in range(5):
        initial_offsets_list.append(base * (1 + rng.normal(0, 0.3, N)))
    
    # Optimize each initial configuration
    for offsets_init in initial_offsets_list:
        if time.time() > deadline - 20:
            break
        try:
            cand_offsets, score = _optimize_offsets(offsets_init, deadline)
            if score > best_score:
                best_score = score
                best_points = _offsets_to_points(cand_offsets)
        except Exception:
            continue
    
    # Perturbation restarts from best
    if best_points is not None and time.time() < deadline - 30:
        best_offsets = None
        # Recover offsets from best points by fitting
        # Actually, let's just track the best offsets
        # Re-run: find best offsets again
        # For now, do random restarts
        base = _regular_offsets()
        for i in range(10):
            if time.time() > deadline - 10:
                break
            noise_scale = 0.05 * (1 - i / 10.0) + 0.01
            perturbed = base + noise_scale * rng.normal(0, 1, N)
            try:
                cand_offsets, score = _optimize_offsets(perturbed, deadline)
                if score > best_score:
                    best_score = score
                    best_points = _offsets_to_points(cand_offsets)
            except Exception:
                continue
    
    # Final polish
    if best_points is not None and time.time() < deadline - 5:
        # Need to recover offsets from points for polishing
        # Fit offsets: for each side k, d_k = n_k . vertex_k (approximately)
        # More precisely, d_k should be such that all points satisfy n_k . x <= d_k
        # and vertex k-1 and k lie on the line n_k . x = d_k
        fitted_offsets = np.zeros(N)
        for k in range(N):
            # The two adjacent vertices to side k are vertices k-1 and k
            # d_k = n_k . vertex_k = n_k . vertex_{k-1} (they should be equal for a valid polygon)
            v_prev = best_points[(k - 1) % N]
            v_curr = best_points[k]
            d_prev = _NORM_X[k] * v_prev[0] + _NORM_Y[k] * v_prev[1]
            d_curr = _NORM_X[k] * v_curr[0] + _NORM_Y[k] * v_curr[1]
            fitted_offsets[k] = (d_prev + d_curr) / 2.0
        
        try:
            polished_offsets, polished_score = _polish_offsets(fitted_offsets, deadline)
            if polished_score > best_score:
                best_score = polished_score
                best_points = _offsets_to_points(polished_offsets)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
