import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14

def _precompute_triples(n):
    """Precompute all (i,j,k) triples with i<j<k."""
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
    return np.array(triples, dtype=int)

TRIPLES = _precompute_triples(N)

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area using precomputed triples."""
    n = len(points)
    if n < 3:
        return 0.0
    
    pi = points[TRIPLES[:, 0]]
    pj = points[TRIPLES[:, 1]]
    pk = points[TRIPLES[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _soft_min_score(points, beta=30.0):
    """
    Smooth surrogate: soft_min of triangle areas / hull area.
    Returns a smooth value that approximates min_area/hull_area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    pi = points[TRIPLES[:, 0]]
    pj = points[TRIPLES[:, 1]]
    pk = points[TRIPLES[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft-min via log-sum-exp: soft_min(a) ≈ (1/beta) * log(sum(exp(-beta * a)))
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    soft_min_val = (1.0 / beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    return soft_min_val / hull_area

def _get_fourier_points(params):
    """
    Generate 14 points using a Fourier series for the radius.
    params: [a1, b1, a2, b2, a3]
    r_k = 1 + a1*cos(theta_k) + b1*sin(theta_k) + a2*cos(2*theta_k) + b2*sin(2*theta_k) + a3*cos(3*theta_k)
    """
    a1, b1, a2, b2, a3 = params
    
    k = np.arange(N)
    theta = 2 * np.pi * k / N
    
    r = (1.0 + a1 * np.cos(theta) + b1 * np.sin(theta) + 
         a2 * np.cos(2 * theta) + b2 * np.sin(2 * theta) +
         a3 * np.cos(3 * theta))
    
    # Clip to ensure we don't hit the origin or go negative
    r = np.maximum(r, 0.15)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.column_stack((x, y))

def _score_fourier(params):
    """Score for Fourier parameterization."""
    points = _get_fourier_points(params)
    return _score(points)

def _soft_min_fourier(params, beta=30.0):
    """Soft-min score for Fourier parameterization."""
    points = _get_fourier_points(params)
    return _soft_min_score(points, beta=beta)

def _optimize_fourier(deadline):
    """
    Optimize the Fourier parameters.
    """
    # Bounds for a1, b1, a2, b2, a3
    # We want to allow significant deformation but not collapse.
    bounds = [(-0.35, 0.35), (-0.35, 0.35), (-0.25, 0.25), (-0.25, 0.25), (-0.2, 0.2)]
    
    best_score = 0.0
    best_params = None
    best_points = None
    
    # Phase 1: Differential Evolution on Fourier params using soft-min
    # 5 parameters, so DE should be fast.
    popsize = 15
    maxiter = 40
    
    try:
        res = differential_evolution(
            lambda p: -_soft_min_fourier(p, beta=30.0),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=True,
            init='sobol'
        )
        
        best_params = res.x
        best_points = _get_fourier_points(best_params)
        best_score = _score(best_points)
    except Exception as e:
        pass
    
    # Phase 2: Local polish with multiple starts using exact score
    if time.time() < deadline - 15.0 and best_params is not None:
        # Start from the DE result and some perturbations
        starts = [best_params.copy()]
        for dim in range(5):
            for delta in [-0.05, 0.05]:
                p = best_params.copy()
                p[dim] += delta
                starts.append(p)
        
        for start in starts:
            if time.time() > deadline - 15.0:
                break
            try:
                res_nm = minimize(
                    lambda p: -_score_fourier(p),
                    start,
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 150}
                )
                if -res_nm.fun > best_score:
                    best_score = -res_nm.fun
                    best_params = res_nm.x
                    best_points = _get_fourier_points(best_params)
            except Exception:
                pass
                
    # Phase 3: Coordinate-level polish to escape Fourier parameterization
    if best_points is not None and time.time() < deadline - 10.0:
        # Use soft-min for smooth gradient, then verify with exact min
        def neg_soft_score(x):
            pts = x.reshape(-1, 2)
            return _soft_min_score(pts, beta=30.0)
        
        try:
            res = minimize(
                neg_soft_score,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            polished = res.x.reshape(-1, 2)
            polished_score = _score(polished)
            
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished
                
        except Exception:
            pass
            
        # Second polish with higher beta
        if time.time() < deadline - 5.0:
            def neg_soft_score_high(x):
                pts = x.reshape(-1, 2)
                return _soft_min_score(pts, beta=100.0)
            
            try:
                res2 = minimize(
                    neg_soft_score_high,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                polished2 = res2.reshape(-1, 2)
                polished2_score = _score(polished2)
                
                if polished2_score > best_score:
                    best_score = polished2_score
                    best_points = polished2
                    
            except Exception:
                pass
        
        # Final Nelder-Mead on coordinates
        if time.time() < deadline - 3.0:
            try:
                res3 = minimize(
                    lambda x: -_score(x.reshape(-1, 2)),
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 100}
                )
                polished3 = res3.x.reshape(-1, 2)
                polished3_score = _score(polished3)
                
                if polished3_score > best_score:
                    best_score = polished3_score
                    best_points = polished3
                    
            except Exception:
                pass
                
    return best_score, best_points

def _build_ring_config(radii, angles_per_ring, phase_offsets):
    """
    Build point set from concentric rings.
    radii: array of ring radii (outermost first)
    angles_per_ring: list of counts per ring
    phase_offsets: list of angular offsets per ring
    """
    pts = []
    for r, n_ang, phase in zip(radii, angles_per_ring, phase_offsets):
        angles = np.linspace(0, 2 * np.pi, n_ang, endpoint=False) + phase
        ring_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        pts.append(ring_pts)
    return np.vstack(pts)

def _optimize_ring_config(n1, n2, n3, r_init, phase_init, time_limit=8.0):
    """
    Optimize a 3-ring configuration with n1+n2+n3=N points.
    Parameters: [r1, r2, r3, phase1, phase2, phase3]
    r1 >= r2 >= r3 > 0, phases in [0, 2*pi)
    """
    start_time = time.time()
    
    def build_from_params(x):
        r1, r2, r3, p1, p2, p3 = x
        r1 = max(0.5, r1)
        r2 = max(0.1, min(r1, r2))
        r3 = max(0.01, min(r2, r3))
        p1 = p1 % (2 * np.pi)
        p2 = p2 % (2 * np.pi)
        p3 = p3 % (2 * np.pi)
        return _build_ring_config([r1, r2, r3], [n1, n2, n3], [p1, p2, p3])
    
    def neg_score_params(x):
        pts = build_from_params(x)
        return -_score(pts)
    
    x0 = np.array([r_init[0], r_init[1], r_init[2], phase_init[0], phase_init[1], phase_init[2]])
    best_score = _score(build_from_params(x0))
    best_x = x0.copy()
    
    bounds = [(0.5, 2.0), (0.1, 2.0), (0.01, 2.0), (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
    
    # Try multiple local optimizations from different starts
    starts = [x0]
    rng = np.random.RandomState(42 + n1*100 + n2*10 + n3)
    for _ in range(4):
        x_pert = x0 + rng.normal(0, 0.15, size=6)
        x_pert[0] = max(0.5, x_pert[0])
        x_pert[1] = max(0.1, min(x_pert[1], x_pert[0]))
        x_pert[2] = max(0.01, min(x_pert[2], x_pert[1]))
        x_pert[3:] = x_pert[3:] % (2*np.pi)
        starts.append(x_pert)
    
    for xi in starts:
        if time.time() > start_time + time_limit - 2:
            break
        try:
            res = minimize(
                neg_score_params, xi, method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 250, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    best_pts = build_from_params(best_x)
    return best_score, best_pts

def _optimize_two_ring_config(n1, n2, r_init, phase_init, time_limit=8.0):
    """
    Optimize a 2-ring configuration with n1+n2=N points.
    Parameters: [r1, r2, phase1, phase2]
    """
    start_time = time.time()
    
    def build_from_params(x):
        r1, r2, p1, p2 = x
        r1 = max(0.5, r1)
        r2 = max(0.05, min(r1, r2))
        p1 = p1 % (2 * np.pi)
        p2 = p2 % (2 * np.pi)
        return _build_ring_config([r1, r2], [n1, n2], [p1, p2])
    
    def neg_score_params(x):
        pts = build_from_params(x)
        return -_score(pts)
    
    x0 = np.array([r_init[0], r_init[1], phase_init[0], phase_init[1]])
    best_score = _score(build_from_params(x0))
    best_x = x0.copy()
    
    bounds = [(0.5, 2.0), (0.05, 2.0), (0, 2*np.pi), (0, 2*np.pi)]
    
    starts = [x0]
    rng = np.random.RandomState(42 + n1*100 + n2)
    for _ in range(5):
        x_pert = x0 + rng.normal(0, 0.15, size=4)
        x_pert[0] = max(0.5, x_pert[0])
        x_pert[1] = max(0.05, min(x_pert[1], x_pert[0]))
        x_pert[2:] = x_pert[2:] % (2*np.pi)
        starts.append(x_pert)
    
    for xi in starts:
        if time.time() > start_time + time_limit - 2:
            break
        try:
            res = minimize(
                neg_score_params, xi, method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 250, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    best_pts = build_from_params(best_x)
    return best_score, best_pts

def _local_optimize_full(points, time_limit=15.0):
    """Full 28D Nelder-Mead refinement as fallback."""
    start_time = time.time()
    x0 = points.flatten()
    
    def obj(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    best_score = _score(points)
    best_pts = points.copy()
    
    try:
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if time.time() - start_time < time_limit:
            if res.fun < -best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(-1, 2).copy()
    except Exception:
        pass
        
    return best_score, best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Strategy 1: Fourier series parameterization (from inspiration)
    # This allows asymmetric deformations that break regular polygon symmetry
    s_fourier, p_fourier = _optimize_fourier(deadline)
    if s_fourier > best_overall_score:
        best_overall_score = s_fourier
        best_overall_points = p_fourier
    
    # Strategy 2: Explore symmetry-reduced ring configurations
    ring_configs_3 = [
        (6, 5, 3),
        (5, 5, 4),
        (7, 4, 3),
        (5, 4, 5),
        (4, 5, 5),
        (6, 4, 4),
        (4, 6, 4),
        (7, 5, 2),
        (2, 6, 6),
    ]
    
    ring_configs_2 = [
        (7, 7),
        (6, 8),
        (8, 6),
        (5, 9),
        (9, 5),
    ]
    
    base_r = [1.0, 0.6, 0.3]
    base_phases = [0.0, np.pi/7, np.pi/5]
    
    # Try 3-ring configs
    for (n1, n2, n3) in ring_configs_3:
        if time.time() > deadline - 45:
            break
        
        for r_ratio in [0.5, 0.6, 0.7, 0.8]:
            if time.time() > deadline - 45:
                break
            r_init = [1.0, 0.5 + 0.3*r_ratio, 0.2 + 0.3*r_ratio]
            r_init[1] = min(r_init[1], r_init[0] * 0.9)
            r_init[2] = min(r_init[2], r_init[1] * 0.9)
            
            for phase_shift in [0.0, np.pi/(2*n1), np.pi/n1]:
                if time.time() > deadline - 45:
                    break
                phase_init = [0.0, phase_shift, 2*phase_shift]
                
                s, p = _optimize_ring_config(n1, n2, n3, r_init, phase_init, time_limit=6.0)
                if s > best_overall_score:
                    best_overall_score = s
                    best_overall_points = p
    
    # Try 2-ring configs
    for (n1, n2) in ring_configs_2:
        if time.time() > deadline - 30:
            break
        
        for r_ratio in [0.4, 0.5, 0.6, 0.7, 0.8]:
            if time.time() > deadline - 30:
                break
            r_init = [1.0, 0.3 + 0.6*r_ratio]
            r_init[1] = min(r_init[1], r_init[0] * 0.95)
            
            for phase_shift in [0.0, np.pi/(2*n1), np.pi/n1, np.pi/(2*n2)]:
                if time.time() > deadline - 30:
                    break
                phase_init = [0.0, phase_shift]
                
                s, p = _optimize_two_ring_config(n1, n2, r_init, phase_init, time_limit=5.0)
                if s > best_overall_score:
                    best_overall_score = s
                    best_overall_points = p
    
    # Final: full-space local optimization from best found
    if best_overall_points is not None and time.time() < deadline - 20:
        s, p = _local_optimize_full(best_overall_points, time_limit=18.0)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_points = p
    
    # Fallback: regular polygon
    if best_overall_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_overall_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_overall_score = _score(best_overall_points)
    
    return best_overall_points
