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

def _get_triangle_areas(points):
    """
    Vectorized calculation of areas for all triples.
    points: (N, 2) array
    returns: array of shape (num_triangles,)
    """
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

def _softmin_score(points, alpha):
    """
    Smooth approximation of the score using softmin of triangle areas.
    This provides a differentiable-like landscape for optimization.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    
    # Softmin: -1/alpha * log(sum(exp(-alpha * areas)))
    a_min = np.min(areas)
    exp_terms = np.exp(-alpha * (areas - a_min))
    log_sum = np.log(np.sum(exp_terms))
    softmin_val = a_min - np.log(np.sum(exp_terms)) / alpha
    
    return softmin_val / hull_area

def _neg_softmin_score(params, alpha):
    """Objective for softmin-based optimization."""
    points = params.reshape(N, 2)
    s = _softmin_score(points, alpha)
    return -s

def _neg_score(params):
    """Objective function for minimization (maximizing score)."""
    points = params.reshape(N, 2)
    s = _score(points)
    return -s

def _make_ring_config(n_points, radius, phase_offset=0.0, n_rings=1, ring_radii=None):
    """Generate a ring-based initial configuration."""
    if ring_radii is None:
        if n_rings == 1:
            ring_radii = [radius]
        else:
            ring_radii = np.linspace(0.3, radius, n_rings).tolist()
    
    points = []
    per_ring = n_points // n_rings
    remainder = n_points % n_rings
    
    for ring_idx, r in enumerate(ring_radii):
        count = per_ring + (1 if ring_idx < remainder else 0)
        for i in range(count):
            angle = 2 * np.pi * i / count + phase_offset + ring_idx * 0.3
            points.append([r * np.cos(angle), r * np.sin(angle)])
    
    while len(points) < n_points:
        idx = len(points)
        angle = 2 * np.pi * idx / n_points + phase_offset
        points.append([radius * 0.5 * np.cos(angle), radius * 0.5 * np.sin(angle)])
    
    return np.array(points[:n_points])

def _optimize_from_start(start_pts, deadline, rng):
    """
    Multi-stage optimization from a starting point:
    1. Softmin-based Nelder-Mead with decreasing alpha
    2. L-BFGS-B polish
    3. Nelder-Mead on hard score
    Returns best points and score found.
    """
    best_pts = start_pts.copy()
    best_score = _score(start_pts)
    
    # Stage 1: Softmin optimization with annealing alpha
    alphas = [100, 50, 20, 10, 5, 2]
    for alpha in alphas:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                _neg_softmin_score,
                best_pts.flatten(),
                args=(alpha,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Stage 2: L-BFGS-B polish with softmin
    if time.time() < deadline - 15:
        try:
            res = minimize(
                _neg_softmin_score,
                best_pts.flatten(),
                args=(2.0,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Stage 3: Nelder-Mead on hard score
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    return best_pts, best_score

def _polish_with_local_search(points, deadline):
    """
    Final high-precision polish: coordinate descent on the hard objective.
    Iteratively move each point to maximize the minimum triangle area.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    
    # Coordinate descent: for each point, try small perturbations
    for outer_iter in range(50):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            
            orig = best_pts[i].copy()
            best_local = orig
            best_local_score = best_score
            
            # Try perturbations in x and y directions
            for dx in [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]:
                for dy in [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]:
                    cand = best_pts.copy()
                    cand[i] = orig + np.array([dx, dy])
                    s = _score(cand)
                    if s > best_local_score + 1e-12:
                        best_local_score = s
                        best_local = orig + np.array([dx, dy])
            
            if np.any(best_local != orig):
                best_pts[i] = best_local
                improved = True
        
        if not improved:
            break
        
        current_score = _score(best_pts)
        if current_score > best_score:
            best_score = current_score
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s kill
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Ring configurations with various parameters
    for radius in [1.0, 1.2, 0.8]:
        for phase in [0.0, np.pi / N, np.pi / (2 * N)]:
            initial_configs.append(_make_ring_config(N, radius, phase_offset=phase))
    
    # Two-ring structures
    for r1, r2 in [(0.5, 1.0), (0.4, 1.1), (0.6, 0.9)]:
        for phase in [0.0, np.pi / N]:
            initial_configs.append(_make_ring_config(N, 1.0, phase_offset=phase, n_rings=2, ring_radii=[r1, r2]))
    
    # Radial perturbation of regular polygon
    for amp in [0.1, 0.2, 0.3]:
        for freq in [2, 3, 5]:
            angles = 2 * np.pi * np.arange(N) / N
            radii = 1.0 + amp * np.sin(freq * angles)
            initial_configs.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # Random configurations
    for _ in range(10):
        initial_configs.append(rng.uniform(-1, 1, (N, 2)))
    
    # Try each initial configuration with multi-stage optimization
    for config in initial_configs:
        if time.time() > deadline - 20:
            break
        
        try:
            cand_pts, score = _optimize_from_start(config, deadline, rng)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            continue
    
    # If we have a good candidate, do perturbation-based restarts
    if best_points is not None and time.time() < deadline - 30:
        for i in range(15):
            if time.time() > deadline - 10:
                break
            
            # Perturb the best solution
            noise_scale = 0.1 * (1 - i / 15.0) + 0.01
            perturbed = best_points + noise_scale * rng.normal(0, 1, (N, 2))
            
            try:
                cand_pts, score = _optimize_from_start(perturbed, deadline, rng)
                if score > best_score:
                    best_score = score
                    best_points = cand_pts
            except Exception:
                continue
    
    # Final high-precision polish using coordinate descent
    if best_points is not None and time.time() < deadline - 5:
        try:
            polished_pts, polished_score = _polish_with_local_search(best_points, deadline)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_pts
        except Exception:
            pass
    
    # Fallback to regular polygon if everything failed
    if best_points is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
