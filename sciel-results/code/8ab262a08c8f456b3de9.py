import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13
NUM_TRIS = N * (N - 1) * (N - 2) // 6

# Pre-compute triangle indices
_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_I_IDX = _TRI_INDICES[:, 0]
_J_IDX = _TRI_INDICES[:, 1]
_K_IDX = _TRI_INDICES[:, 2]

def _get_triangle_areas(points):
    """Vectorized calculation of areas for all triples."""
    pi = points[_I_IDX]
    pj = points[_J_IDX]
    pk = points[_K_IDX]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

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

def _neg_score(params):
    """Objective function for minimization (maximizing score)."""
    points = params.reshape(N, 2)
    s = _score(points)
    return -s

def _softmin_score(points, alpha):
    """Smooth approximation of the score using softmin of triangle areas."""
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

def _fourier_to_points(params):
    """
    Fourier parameterization of point positions.
    params: [c0x, c0y, r1, theta1, r2, theta2, ..., rK, thetaK]
    where c0 is the center, and (r_k, theta_k) are Fourier coefficients.
    K = N//2 = 6 for N=13.
    Total params = 2 + 2*K = 14.
    
    Point i: p_i = c0 + sum_{k=1}^{K} r_k * [cos(k*2*pi*i/N + theta_k), sin(k*2*pi*i/N + theta_k)]
    """
    K = N // 2  # 6
    c0x, c0y = params[0], params[1]
    i = np.arange(N)
    x = np.full(N, c0x)
    y = np.full(N, c0y)
    for k in range(1, K + 1):
        r_k = params[2 + 2 * (k - 1)]
        theta_k = params[3 + 2 * (k - 1)]
        angle = k * 2.0 * np.pi * i / N + theta_k
        x += r_k * np.cos(angle)
        y += r_k * np.sin(angle)
    return np.column_stack([x, y])

def _neg_fourier_score(params):
    """Objective for Fourier parameterization optimization."""
    points = _fourier_to_points(params)
    s = _score(points)
    return -s

def _neg_fourier_softmin(params, alpha):
    """Softmin objective for Fourier parameterization."""
    points = _fourier_to_points(params)
    s = _softmin_score(points, alpha)
    return -s

def _optimize_fourier(start_params, deadline, rng):
    """
    Optimize in Fourier parameter space.
    Uses softmin annealing followed by hard score polish.
    """
    best_params = start_params.copy()
    best_score = _score(_fourier_to_points(best_params))
    
    # Stage 1: Softmin annealing in Fourier space
    alphas = [50, 20, 10, 5, 2, 1]
    for alpha in alphas:
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(
                _neg_fourier_softmin,
                best_params,
                args=(alpha,),
                method='Nelder-Mead',
                options={'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            cand_params = res.x
            cand_pts = _fourier_to_points(cand_params)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_params = cand_params
        except Exception:
            pass
    
    # Stage 2: Hard score Nelder-Mead in Fourier space
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _neg_fourier_score,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            cand_params = res.x
            cand_pts = _fourier_to_points(cand_params)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_params = cand_params
        except Exception:
            pass
    
    best_pts = _fourier_to_points(best_params)
    return best_pts, best_score

def _optimize_from_start(start_pts, deadline, rng):
    """
    Multi-stage optimization from a starting point in Cartesian space:
    1. Softmin-based Nelder-Mead with decreasing alpha
    2. L-BFGS-B polish
    3. Nelder-Mead on hard score
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
                options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Stage 2: L-BFGS-B polish with softmin
    for alpha_polish in [2.0, 1.0, 0.5]:
        if time.time() < deadline - 10:
            try:
                res = minimize(
                    _neg_softmin_score,
                    best_pts.flatten(),
                    args=(alpha_polish,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
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
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    return best_pts, best_score

def _coordinate_local_search(points, deadline, rng, initial_step=0.05, min_step=1e-6, max_iter=200):
    """
    Coordinate-wise pattern search on the hard score.
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
        n_iter += 1
        if improved:
            step *= 1.2
        else:
            step *= 0.5
    
    return best_pts, best_score

def _polish_with_local_search(points, deadline, rng):
    """
    Multi-stage polishing: coordinate search + Nelder-Mead + softmin polish.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    
    # Stage 1: Coordinate-wise local search
    if time.time() < deadline - 10:
        cand_pts, score = _coordinate_local_search(best_pts, deadline, rng)
        if score > best_score:
            best_score = score
            best_pts = cand_pts
    
    # Stage 2: Nelder-Mead on hard score
    if time.time() < deadline - 8:
        try:
            res = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Stage 3: Fine coordinate search
    if time.time() < deadline - 5:
        cand_pts, score = _coordinate_local_search(best_pts, deadline, rng, initial_step=0.01)
        if score > best_score:
            best_score = score
            best_pts = cand_pts
    
    # Stage 4: Ultra-fine Nelder-Mead
    if time.time() < deadline - 3:
        try:
            res = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
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
    deadline = start_time + 110.0
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)
    
    # === PHASE 1: Fourier parameterization optimization ===
    # Initialize Fourier params from a regular polygon
    K = N // 2
    # Regular polygon: only r_1 is non-zero, with r_1 = 1, theta_1 = 0
    fourier_init = np.zeros(2 + 2 * K)
    fourier_init[2] = 1.0  # r_1
    fourier_init[3] = 0.0  # theta_1
    
    # Try multiple Fourier initializations
    fourier_inits = []
    # Regular polygon
    fourier_inits.append(fourier_init.copy())
    
    # Perturbed regular polygon
    for trial in range(5):
        fi = fourier_init.copy()
        fi[2] = 1.0 + rng.uniform(-0.1, 0.1)
        fi[3] = rng.uniform(-0.1, 0.1)
        # Add small higher harmonics
        for k in range(2, K + 1):
            fi[2 + 2 * (k - 1)] = rng.uniform(0, 0.15)
            fi[3 + 2 * (k - 1)] = rng.uniform(0, 2 * np.pi)
        fourier_inits.append(fi)
    
    # More diverse Fourier inits
    for trial in range(5):
        fi = np.zeros(2 + 2 * K)
        fi[0] = rng.uniform(-0.2, 0.2)
        fi[1] = rng.uniform(-0.2, 0.2)
        fi[2] = rng.uniform(0.7, 1.3)
        fi[3] = rng.uniform(0, 2 * np.pi)
        for k in range(2, K + 1):
            fi[2 + 2 * (k - 1)] = rng.uniform(0, 0.2)
            fi[3 + 2 * (k - 1)] = rng.uniform(0, 2 * np.pi)
        fourier_inits.append(fi)
    
    # Optimize each Fourier init
    for fi in fourier_inits:
        if time.time() > deadline - 30:
            break
        try:
            cand_pts, score = _optimize_fourier(fi, deadline, rng)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            continue
    
    # === PHASE 2: Cartesian-space diverse initial configs ===
    initial_configs = []
    
    # Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Slightly perturbed regular polygons
    for perturb in [0.02, 0.05, 0.1]:
        angles_p = 2 * np.pi * np.arange(N) / N + rng.uniform(-perturb, perturb, N)
        radii_p = 1.0 + rng.uniform(-perturb, perturb, N)
        initial_configs.append(np.column_stack([radii_p * np.cos(angles_p), radii_p * np.sin(angles_p)]))
    
    # Ring configurations
    for radius in [1.0, 1.2, 0.8]:
        for phase in [0.0, np.pi / N, np.pi / (2 * N)]:
            initial_configs.append(_make_ring_config(N, radius, phase_offset=phase))
    
    # Two-ring structures
    for r1, r2 in [(0.5, 1.0), (0.4, 1.1), (0.6, 0.9)]:
        for phase in [0.0, np.pi / N]:
            initial_configs.append(_make_ring_config(N, 1.0, phase_offset=phase, n_rings=2, ring_radii=[r1, r2]))
    
    # Radial perturbation
    for amp in [0.1, 0.2, 0.3]:
        for freq in [2, 3, 5]:
            angles = 2 * np.pi * np.arange(N) / N
            radii = 1.0 + amp * np.sin(freq * angles)
            initial_configs.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # Random configurations
    for _ in range(5):
        initial_configs.append(rng.uniform(-1, 1, (N, 2)))
    
    # Add best Fourier result
    if best_points is not None:
        initial_configs.append(best_points)
    
    # Sort by initial score
    scored_configs = []
    for config in initial_configs:
        try:
            s = _score(config)
            scored_configs.append((s, config))
        except Exception:
            pass
    scored_configs.sort(key=lambda x: -x[0])
    
    # Optimize top configs
    for s_init, config in scored_configs:
        if time.time() > deadline - 25:
            break
        try:
            cand_pts, score = _optimize_from_start(config, deadline, rng)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            continue
    
    # === PHASE 3: Polish best result ===
    if best_points is not None and time.time() < deadline - 20:
        cand_pts, score = _polish_with_local_search(best_points, deadline, rng)
        if score > best_score:
            best_score = score
            best_points = cand_pts
    
    # === PHASE 4: Perturbation restarts from best ===
    if best_points is not None and time.time() < deadline - 15:
        n_restarts = 0
        while time.time() < deadline - 10:
            n_restarts += 1
            frac = min(n_restarts / 15.0, 1.0)
            noise_scale = 0.1 * (1.0 - frac) + 0.003
            perturbed = best_points + noise_scale * rng.normal(0, 1, (N, 2))
            try:
                cand_pts, score = _polish_with_local_search(perturbed, deadline, rng)
                if score > best_score:
                    best_score = score
                    best_points = cand_pts
            except Exception:
                continue
    
    # Fallback
    if best_points is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
