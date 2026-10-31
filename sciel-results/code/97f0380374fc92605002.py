import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _soft_min_value(areas, p):
    """Numerically stable soft-min of array."""
    areas = np.maximum(areas, 1e-15)
    neg_p_areas = -p * areas
    max_neg = np.max(neg_p_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_p_areas - max_neg)))
    return -log_sum / p

def _soft_score(points, p=50):
    """Compute soft score directly from points."""
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    if np.allclose(pts, 0):
        return 0.0
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(pts)
    soft_min = _soft_min_value(areas, p)
    return soft_min / hull_area

def _neg_soft_score(x_flat, p=50):
    """Objective: minimize negative soft score."""
    points = x_flat.reshape(N, 2)
    points = points - np.mean(points, axis=0)
    s = _soft_score(points, p)
    if s <= 0:
        return 1e10
    return -s

def _neg_score(x_flat):
    """Objective: minimize negative hard score."""
    points = x_flat.reshape(N, 2)
    points = points - np.mean(points, axis=0)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s

def _make_ring_config(n_outer, n_inner, r_inner, angle_offset):
    """Create a two-ring configuration."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_inner, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([outer_pts, inner_pts])

def _annealed_soft_minimize(x0, deadline, p_start=20, p_end=200, n_phases=6):
    """Annealed soft-minimize: gradually increase p while optimizing."""
    best_x = x0.copy()
    best_s = _score(x0.reshape(N, 2))
    
    p_values = np.geomspace(p_start, p_end, n_phases)
    
    for idx, p in enumerate(p_values):
        if time.time() > deadline - 3:
            break
        
        # Use Nelder-Mead on the soft objective for this p
        try:
            res = minimize(
                lambda x, p_val=p: _neg_soft_score(x, p_val),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final polish with hard objective
    if time.time() < deadline - 1:
        try:
            res_final = minimize(
                _neg_score,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            s_final = _score(res_final.x.reshape(N, 2))
            if s_final > best_s:
                best_s = s_final
                best_x = res_final.x.copy()
        except Exception:
            pass
    
    return best_x.reshape(N, 2), best_s

def _local_polish(points, deadline):
    """Multi-phase local optimization from a starting point using annealed soft-min."""
    best_x = points.flatten().copy()
    best_s = _score(points)
    
    # Phase 1: Annealed soft-minimize (main optimization)
    if time.time() < deadline - 5:
        try:
            pts_opt, s_opt = _annealed_soft_minimize(best_x, deadline - 2)
            if s_opt > best_s:
                best_s = s_opt
                best_x = pts_opt.flatten().copy()
        except Exception:
            pass
    
    # Phase 2: Additional Nelder-Mead refinement on hard objective
    if time.time() < deadline - 1:
        try:
            res = minimize(
                _neg_score,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x.reshape(N, 2), best_s

def _generate_inits(rng):
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]).flatten())
    
    # 2. Two-ring configurations
    for n_out in [5, 6, 7, 8]:
        n_in = N - n_out
        if n_in < 3:
            continue
        for r_in in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
            for offset in [0, np.pi / n_out, np.pi / (2 * n_out), np.pi / (3 * n_out)]:
                pts = _make_ring_config(n_out, n_in, r_in, offset)
                inits.append(pts.flatten())
    
    # 3. Three-ring configurations
    for n1, n2, n3 in [(4, 4, 5), (5, 4, 4), (6, 5, 2), (7, 3, 3), (5, 5, 3)]:
        for r2 in [0.4, 0.5, 0.6, 0.7]:
            for r3 in [0.15, 0.25, 0.35, 0.45]:
                a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
                a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n1
                a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + np.pi / n2
                pts = np.vstack([
                    np.column_stack([np.cos(a1), np.sin(a1)]),
                    np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
                    np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
                ])
                inits.append(pts.flatten())
    
    # 4. Random starts
    for _ in range(15):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts.flatten())
    
    # 5. Perturbed regular polygon
    for noise_scale in [0.01, 0.03, 0.05, 0.08, 0.12, 0.18]:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0 + noise_scale * (rng.random(N) - 0.5)
        angles_perturbed = angles + noise_scale * (rng.random(N) - 0.5)
        pts = np.column_stack([radii * np.cos(angles_perturbed), radii * np.sin(angles_perturbed)])
        inits.append(pts.flatten())
    
    # 6. Quasi-regular with alternating radii
    for base_r in [0.8, 0.9, 1.0]:
        for alt_r in [0.5, 0.6, 0.7]:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            radii = np.where(np.arange(N) % 2 == 0, base_r, alt_r)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            inits.append(pts.flatten())
    
    # 7. Fibonacci spiral-inspired distribution
    golden_angle = np.pi * (3 - np.sqrt(5))
    for scale in [0.8, 1.0, 1.2]:
        pts = []
        for i in range(N):
            r = scale * np.sqrt(i / N)
            theta = i * golden_angle
            pts.append([r * np.cos(theta), r * np.sin(theta)])
        inits.append(np.array(pts).flatten())
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    rng = np.random.default_rng(12345)
    inits = _generate_inits(rng)
    
    # Shuffle inits
    rng2 = np.random.default_rng(999)
    indices = rng2.permutation(len(inits))
    inits = [inits[i] for i in indices]
    
    # Optimize from each start
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        t_start = time.time()
        pts_opt, s_opt = _local_polish(x0.reshape(N, 2), deadline - 5)
        _update_best(pts_opt)
        
        if i % 5 == 0:
            print(f"Init {i}/{len(inits)}: score={best_score:.6f}, elapsed={time.time()-start_time:.1f}s", flush=True)
    
    # Final high-precision polish on best
    if best_points is not None and time.time() < deadline - 3:
        try:
            res_final = minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            _update_best(res_final.x.reshape(N, 2))
        except Exception:
            pass
    
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
