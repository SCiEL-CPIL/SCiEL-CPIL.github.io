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
    """
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

def _fourier_to_points(params):
    """
    Convert Fourier parameters to 13 points on a perturbed regular polygon.
    params: [r0, a1, b1, a2, b2, cx, cy] (7 params)
    """
    r0 = params[0]
    a1, b1 = params[1], params[2]
    a2, b2 = params[3], params[4]
    cx, cy = params[5], params[6]
    
    angles = 2 * np.pi * np.arange(N) / N
    radii = r0 + a1 * np.cos(angles) + b1 * np.sin(angles) + a2 * np.cos(2 * angles) + b2 * np.sin(2 * angles)
    
    radii = np.maximum(radii, 0.1)
    
    points = np.column_stack([radii * np.cos(angles) + cx, radii * np.sin(angles) + cy])
    return points

def _neg_fourier_score(params, alpha):
    """
    Objective for Fourier-parameterized optimization.
    Uses softmin for smoothness.
    """
    points = _fourier_to_points(params)
    s = _softmin_score(points, alpha)
    return -s

def _neg_fourier_score_hard(params):
    """
    Objective for Fourier-parameterized optimization (hard score).
    """
    points = _fourier_to_points(params)
    s = _score(points)
    return -s

def _optimize_fourier(deadline, rng):
    """
    Optimize using Fourier parameterization.
    This reduces dimensionality from 26 to 7 and enforces smooth convexity.
    """
    best_score = 0.0
    best_params = None
    best_points = None
    
    # Initial parameter guesses
    initial_params_list = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Regular polygon
        [1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0],
        [0.9, 0.03, 0.02, 0.02, 0.01, 0.0, 0.0],
        [1.1, -0.02, 0.03, 0.01, -0.02, 0.0, 0.0],
    ]
    
    # Add random initializations
    for _ in range(5):
        p = [1.0 + rng.normal(0, 0.1)]
        for _ in range(4):
            p.append(rng.normal(0, 0.05))
        p.extend([rng.normal(0, 0.05), rng.normal(0, 0.05)])
        initial_params_list.append(p)
    
    for init_params in initial_params_list:
        if time.time() > deadline - 15:
            break
        
        init_params = np.array(init_params, dtype=float)
        
        # Multi-stage optimization in Fourier space
        current_params = init_params.copy()
        current_score = _score(_fourier_to_points(current_params))
        
        # Stage 1: Softmin with various alphas
        for alpha in [50, 20, 10, 5]:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    _neg_fourier_score,
                    current_params,
                    args=(alpha,),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
                )
                cand_params = res.x
                cand_score = _score(_fourier_to_points(cand_params))
                if cand_score > current_score:
                    current_score = cand_score
                    current_params = cand_params
            except Exception:
                pass
        
        # Stage 2: L-BFGS-B polish
        if time.time() < deadline - 8:
            try:
                res = minimize(
                    _neg_fourier_score,
                    current_params,
                    args=(5.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                cand_params = res.x
                cand_score = _score(_fourier_to_points(cand_params))
                if cand_score > current_score:
                    current_score = cand_score
                    current_params = cand_params
            except Exception:
                pass
        
        # Stage 3: Hard score Nelder-Mead
        if time.time() < deadline - 5:
            try:
                res = minimize(
                    _neg_fourier_score_hard,
                    current_params,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                cand_params = res.x
                cand_score = _score(_fourier_to_points(cand_params))
                if cand_score > current_score:
                    current_score = cand_score
                    current_params = cand_params
            except Exception:
                pass
        
        # Update global best
        if current_score > best_score:
            best_score = current_score
            best_params = current_params.copy()
            best_points = _fourier_to_points(current_params)
    
    return best_points, best_score

def _coordinate_local_search(points, deadline, rng, initial_step=0.05, min_step=1e-6, max_iter=200):
    """
    Coordinate-wise pattern search on the hard score.
    Tries moving each coordinate in +/- direction, accepting improvements.
    Adapts step size based on success/failure.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    step = initial_step
    n_iter = 0
    
    while time.time() < deadline - 5 and step > min_step and n_iter < max_iter:
        improved = False
        for i in range(N):
            for dim in range(2):
                for direction in [1, -1]:
                    trial = best_pts.copy()
                    trial[i, dim] += direction * step
                    trial_score = _score(trial)
                    if trial_score > best_score:
                        best_pts = trial
                        best_score = trial_score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        n_iter += 1
        if not improved:
            step *= 0.5
        else:
            step = min(step * 1.2, initial_step)
    
    return best_pts, best_score

def _optimize_from_start(start_pts, deadline, rng):
    """
    Multi-stage optimization from a starting point:
    1. Softmin-based Nelder-Mead with decreasing alpha
    2. L-BFGS-B polish
    3. Coordinate-wise local search on hard score
    4. Ultra-fine Nelder-Mead polish
    Returns best points and score found.
    """
    best_pts = start_pts.copy()
    best_score = _score(start_pts)
    
    # Stage 1: Softmin optimization with annealing alpha
    alphas = [100, 50, 20, 10, 5, 2]
    for alpha in alphas:
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(
                _neg_softmin_score,
                best_pts.flatten(),
                args=(alpha,),
                method='Nelder-Mead',
                options={'maxiter': 600, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Stage 2: L-BFGS-B polish with softmin
    for alpha_polish in [2.0, 1.0]:
        if time.time() < deadline - 15:
            try:
                res = minimize(
                    _neg_softmin_score,
                    best_pts.flatten(),
                    args=(alpha_polish,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                cand_pts = res.x.reshape(N, 2)
                score = _score(cand_pts)
                if score > best_score:
                    best_score = score
                    best_pts = cand_pts
            except Exception:
                pass
    
    # Stage 3: Coordinate-wise local search on hard score
    if time.time() < deadline - 10:
        try:
            cand_pts, score = _coordinate_local_search(
                best_pts, deadline, rng, initial_step=0.05, min_step=1e-7, max_iter=150
            )
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Stage 4: Ultra-fine Nelder-Mead polish on hard score
    if time.time() < deadline - 3:
        try:
            res = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    return best_pts, best_score

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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s kill
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)
    
    # First, try Fourier optimization which is highly effective
    if time.time() < deadline - 30:
        try:
            fourier_pts, fourier_score = _optimize_fourier(deadline - 30, rng)
            if fourier_pts is not None and fourier_score > best_score:
                best_score = fourier_score
                best_points = fourier_pts
        except Exception:
            pass
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Slightly perturbed regular polygons (near-optimal starts)
    for perturb in [0.02, 0.05, 0.1]:
        angles_p = 2 * np.pi * np.arange(N) / N + rng.uniform(-perturb, perturb, N)
        radii_p = 1.0 + rng.uniform(-perturb, perturb, N)
        initial_configs.append(np.column_stack([radii_p * np.cos(angles_p), radii_p * np.sin(angles_p)]))
    
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
    for _ in range(8):
        initial_configs.append(rng.uniform(-1, 1, (N, 2)))
    
    # If we have a good Fourier result, add it to configs
    if best_points is not None:
        initial_configs.insert(0, best_points.copy())
    
    # Try each initial configuration with multi-stage optimization
    scored_configs = []
    for config in initial_configs:
        try:
            s = _score(config)
            scored_configs.append((s, config))
        except Exception:
            pass
    scored_configs.sort(key=lambda x: -x[0])
    
    # Only try top-5 configs to leave time for restarts
    top_configs = scored_configs[:5]
    
    for s_init, config in top_configs:
        if time.time() > deadline - 40:
            break
        
        try:
            cand_pts, score = _optimize_from_start(config, deadline, rng)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            continue
    
    # Perturbation-based restarts with coordinate search
    if best_points is not None and time.time() < deadline - 30:
        n_restarts = 0
        while time.time() < deadline - 8:
            n_restarts += 1
            frac = min(n_restarts / 15.0, 1.0)
            noise_scale = 0.12 * (1.0 - frac) + 0.003
            perturbed = best_points + noise_scale * rng.normal(0, 1, (N, 2))
            
            try:
                cand_pts, score = _optimize_from_start(perturbed, deadline, rng)
                if score > best_score:
                    best_score = score
                    best_points = cand_pts
            except Exception:
                continue
    
    # Fallback to regular polygon if everything failed
    if best_points is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
