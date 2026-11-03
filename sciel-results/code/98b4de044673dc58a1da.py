"""Combined Heilbronn triangles program: barycentric representation + smooth minimization."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 11
H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Precompute all C(11,3) = 165 triples
_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_N_TRIPLES = _TRIPLES.shape[0]

# Vertices of the equilateral triangle
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, H]])


def _min_triangle_area_fast(points):
    """Fully vectorized min triangle area for N=11 points."""
    p1 = points[_TRIPLES[:, 0]]
    p2 = points[_TRIPLES[:, 1]]
    p3 = points[_TRIPLES[:, 2]]
    d1 = p2 - p1
    d2 = p3 - p1
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle (0,0), (1,0), (0.5, H)."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    # Clip y to [0, H]
    y = np.clip(y, 0, H)
    
    # For a given y, x must be in [y/S3, 1 - y/S3]
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)
    
    points[:, 0] = x
    points[:, 1] = y
    return points


def _bary_to_cart(bary):
    """Convert barycentric coordinates to Cartesian.
    bary: (n, 2) array with columns (u, v) where u+u+v <= 1
    Vertex 0 = (0,0), Vertex 1 = (1,0), Vertex 2 = (0.5, H)
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = H * w
    return np.column_stack([x, y])


def _cart_to_bary(points):
    """Convert Cartesian points to barycentric coordinates (u, v)."""
    x = points[:, 0]
    y = points[:, 1]
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])


def _all_dets_from_bary(bary):
    """
    Compute all 165 determinants (twice the area in barycentric coords).
    bary: (N, 2) array with columns (u, v), w = 1-u-v
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx = _TRIPLES[:, 0]
    j_idx = _TRIPLES[:, 1]
    k_idx = _TRIPLES[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return det


def _min_area_ratio_from_bary(bary):
    """
    Returns the minimum absolute determinant (proportional to area ratio).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
        
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))


def _project_to_bary_simplex(bary):
    """
    Project points to the valid barycentric simplex.
    """
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))


def _objective_smooth(x, beta=100.0, penalty_lambda=100.0):
    """
    Smooth objective for L-BFGS-B.
    x: (2N,) array of u, v coords.
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for invalid barycentric coords (u,v,w < 0)
    penalty = 0.0
    excess_u = np.maximum(0, -u)
    excess_v = np.maximum(0, -v)
    excess_w = np.maximum(0, -w)
    penalty += penalty_lambda * (np.sum(excess_u**2) + np.sum(excess_v**2) + np.sum(excess_w**2))
    
    # Clamp for area computation
    u_c = np.clip(u, 0, None)
    v_c = np.clip(v, 0, None)
    w_c = 1.0 - u_c - v_c
    mask = w_c < 0
    if np.any(mask):
        s = u_c[mask] + v_c[mask]
        if s > 0:
            u_c[mask] /= s
            v_c[mask] /= s
        else:
            u_c[mask] = 1/3
            v_c[mask] = 1/3
    
    bary_clamped = np.column_stack((u_c, v_c))
    dets = _all_dets_from_bary(bary_clamped)
    areas = np.abs(dets)
    
    m = np.min(areas)
    shifted = areas - m
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return -smooth_min + penalty


def _objective_slsqp(x):
    """
    Objective for SLSQP: minimize -min_area.
    x: (2N,) array.
    """
    bary = x.reshape(N, 2)
    u = np.clip(bary[:, 0], 0, None)
    v = np.clip(bary[:, 1], 0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    
    bary_clamped = np.column_stack((u, v))
    dets = _all_dets_from_bary(bary_clamped)
    min_area = np.min(np.abs(dets))
    return -min_area


def _generate_fps_initialization(n_points=N, n_iter=50, seed=42):
    """
    Farthest Point Sampling based initialization.
    """
    rng = np.random.RandomState(seed)
    
    points = [np.array([1/3, 1/3])]
    
    for _ in range(n_points - 1):
        n_candidates = 1000
        r1 = rng.rand(n_candidates)
        r2 = rng.rand(n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        cx = r1 + 0.5 * r2
        cy = H * r2
        candidates = np.column_stack([cx, cy])
        
        pts_arr = np.array(points)
        diff = candidates[:, np.newaxis, :] - pts_arr[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        min_dist_sq = np.min(dist_sq, axis=1)
        best_idx = np.argmax(min_dist_sq)
        points.append(candidates[best_idx])
    
    return np.array(points)


def _structured_inits():
    """Generate structured initial configurations based on symmetry."""
    inits = []
    
    # 1. Symmetric arrangement with center and rings
    center = np.array([0.5, H/3])
    r1 = 0.12
    r2 = 0.22
    r3 = 0.32
    
    pts = [center]
    for i in range(6):
        angle = i * np.pi / 3 + np.pi/6
        x = 0.5 + r1 * np.cos(angle)
        y = H/3 + r1 * np.sin(angle)
        pts.append(np.array([x, y]))
    for i in range(4):
        angle = i * np.pi / 2 + np.pi/4
        x = 0.5 + r2 * np.cos(angle)
        y = H/3 + r2 * np.sin(angle)
        pts.append(np.array([x, y]))
    
    pts = np.array(pts)
    pts = _clip_to_triangle(pts)
    inits.append(pts)
    
    # 2. Corner-emphasized grid
    corner_pts = np.array([
        [0.05, 0.02],
        [0.95, 0.02],
        [0.5, H - 0.02],
        [0.15, 0.10],
        [0.85, 0.10],
        [0.35, H - 0.12],
        [0.65, H - 0.12],
        [0.5, H/3],
        [0.30, H/5],
        [0.70, H/5],
        [0.5, 2*H/5],
    ])
    corner_pts = _clip_to_triangle(corner_pts)
    inits.append(corner_pts)
    
    # 3. Barycentric structured
    bary_configs = [
        [[0.05, 0.05],
         [0.95, 0.00],
         [0.05, 0.95],
         [0.50, 0.50],
         [0.50, 0.00],
         [0.00, 0.50],
         [1/3, 1/3],
         [0.25, 0.25],
         [0.25, 0.50],
         [0.50, 0.25],
         [0.40, 0.30]],
    ]
    for bary in bary_configs:
        bary = np.array(bary)
        bary = _project_to_bary_simplex(bary)
        pts = _bary_to_cart(bary)
        pts = _clip_to_triangle(pts)
        inits.append(pts)
    
    # 4. FPS-based with different seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000]:
        pts = _generate_fps_initialization(n_points=N, seed=seed)
        pts = _clip_to_triangle(pts)
        inits.append(pts)
    
    # 5. Hexagonal lattice pattern
    bary1 = np.array([
        [0.0, 0.0],    # V0
        [1.0, 0.0],    # V1
        [0.0, 1.0],    # V2
        [0.5, 0.0],    # mid V0-V1
        [0.5, 0.5],    # mid V1-V2
        [0.0, 0.5],    # mid V2-V0
        [1/3, 1/3],    # centroid
        [0.25, 0.0],   # 1/4 V0-V1
        [0.75, 0.0],   # 3/4 V0-V1
        [0.25, 0.25],  # interior
        [0.5, 0.75],   # interior
    ])
    bary1 = _project_to_bary_simplex(bary1)
    pts = _bary_to_cart(bary1)
    pts = _clip_to_triangle(pts)
    inits.append(pts)
    
    # 6. Concentric hexagons
    bary2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.0, 0.333],
        [0.0, 0.667],
        [0.333, 0.333],
        [0.667, 0.333],
        [0.333, 0.667],
        [0.5, 0.5],
    ])
    bary2 = _project_to_bary_simplex(bary2)
    pts = _bary_to_cart(bary2)
    pts = _clip_to_triangle(pts)
    inits.append(pts)
    
    # 7. Symmetric grid with jitter
    rng3 = np.random.RandomState(123)
    bary3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.0, 0.25],
        [0.0, 0.75],
        [0.25, 0.25],
        [0.5, 0.25],
        [0.25, 0.5],
        [0.5, 0.5],
    ])
    bary3 += rng3.normal(0, 0.02, size=bary3.shape)
    bary3 = _project_to_bary_simplex(bary3)
    pts = _bary_to_cart(bary3)
    pts = _clip_to_triangle(pts)
    inits.append(pts)
    
    # 8. Random with structured bias
    rng4 = np.random.RandomState(456)
    bary4 = []
    bary4.append([0.0, 0.0])
    bary4.append([1.0, 0.0])
    bary4.append([0.0, 1.0])
    bary4.append([1/3, 1/3])
    for _ in range(7):
        u = rng4.uniform(0.1, 0.9)
        v = rng4.uniform(0.1, 0.9)
        if u + v < 0.85:
            bary4.append([u, v])
        else:
            bary4.append([u * 0.8, v * 0.8])
    bary4 = _project_to_bary_simplex(np.array(bary4[:N]))
    pts = _bary_to_cart(bary4)
    pts = _clip_to_triangle(pts)
    inits.append(pts)
    
    return inits


def _simulated_annealing(points, n_iter=5000, t0=0.05, t_end=0.001, seed=42):
    """
    Simulated Annealing to escape local optima.
    """
    rng = np.random.RandomState(seed)
    current = points.copy()
    current_score = _min_triangle_area_fast(current)
    best = current.copy()
    best_score = current_score
    
    t = t0
    cooling = (t_end / t0) ** (1.0 / n_iter)
    
    for i in range(n_iter):
        t *= cooling
        
        # Generate perturbation
        scale = 0.05 * t / t0
        perturbed = current + rng.randn(N, 2) * scale
        perturbed = _clip_to_triangle(perturbed)
        
        perturbed_score = _min_triangle_area_fast(perturbed)
        delta = perturbed_score - current_score
        
        if delta > 0 or rng.rand() < np.exp(delta / max(t, 1e-10)):
            current = perturbed
            current_score = perturbed_score
            
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
    
    return best, best_score


def _local_optimize_nm(x0, maxiter=300, xatol=1e-8, fatol=1e-8):
    """Run Nelder-Mead local optimization from x0."""
    try:
        res = minimize(
            _objective_slsqp, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0


def _local_optimize_lbfgs(x0, maxiter=200, gtol=1e-9, beta=100.0):
    """Run L-BFGS-B local optimization from x0."""
    try:
        res = minimize(
            lambda x: _objective_smooth(x, beta=beta),
            x0,
            method='L-BFGS-B',
            bounds=[(0, 1), (0, 1)] * N,
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        return res.x
    except Exception:
        return x0


def _local_optimize_slsqp(x0, maxiter=200):
    """Run SLSQP local optimization from x0."""
    try:
        res = minimize(
            _objective_slsqp, x0, method='SLSQP',
            bounds=[(0, 1), (0, 1)] * N,
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return res.x
    except Exception:
        return x0


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    
    inits = _structured_inits()
    
    # Evaluate all initial configurations
    for pts in inits:
        pts = _clip_to_triangle(pts)
        score = _min_triangle_area_fast(pts)
        if score > best_score:
            best_score = score
            best_points = pts
    
    # Phase 1: Differential Evolution (global search)
    de_time_budget = 40
    if time.time() < deadline - de_time_budget:
        try:
            bounds = [(0, 1), (0, H)] * N
            
            def de_objective(flat):
                points = flat.reshape(N, 2)
                points = _clip_to_triangle(points)
                return -_min_triangle_area_fast(points)
            
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=30,
                popsize=15,
                tol=0.005,
                seed=42,
                polish=False,
                init='latinhypercube',
                updating='immediate'
            )
            
            if de_result.fun < 0:
                score = -de_result.fun
                if score > best_score:
                    best_score = score
                    best_points = _clip_to_triangle(de_result.x.reshape(N, 2))
                    
        except Exception:
            pass
    
    # Phase 2: Local refinement with multiple methods on diverse starts
    candidates = []
    if best_points is not None:
        candidates.append(best_points)
    candidates.extend(inits)
    
    # Add perturbations of best point
    if best_points is not None:
        rng = np.random.RandomState(42)
        for _ in range(6):
            perturbed = best_points + rng.randn(N, 2) * 0.06
            perturbed = _clip_to_triangle(perturbed)
            candidates.append(perturbed)
    
    for i, init_pts in enumerate(candidates):
        if time.time() > deadline - 8:
            break
            
        init_points = _clip_to_triangle(init_pts)
        init_bary = _cart_to_bary(init_points)
        
        # SLSQP
        try:
            res_bary = _local_optimize_slsqp(init_bary.flatten(), maxiter=200)
            res_pts = _bary_to_cart(_project_to_bary_simplex(res_bary.reshape(N, 2)))
            res_pts = _clip_to_triangle(res_pts)
            score = _min_triangle_area_fast(res_pts)
            if score > best_score:
                best_score = score
                best_points = res_pts
        except Exception:
            pass
        
        # Nelder-Mead
        if time.time() > deadline - 8:
            break
        try:
            res_bary = _local_optimize_nm(init_bary.flatten(), maxiter=300)
            res_pts = _bary_to_cart(_project_to_bary_simplex(res_bary.reshape(N, 2)))
            res_pts = _clip_to_triangle(res_pts)
            score = _min_triangle_area_fast(res_pts)
            if score > best_score:
                best_score = score
                best_points = res_pts
        except Exception:
            pass
        
        # L-BFGS-B with smooth objective
        if time.time() > deadline - 8:
            break
        try:
            res_bary = _local_optimize_lbfgs(init_bary.flatten(), maxiter=200, beta=100.0)
            res_pts = _bary_to_cart(_project_to_bary_simplex(res_bary.reshape(N, 2)))
            res_pts = _clip_to_triangle(res_pts)
            score = _min_triangle_area_fast(res_pts)
            if score > best_score:
                best_score = score
                best_points = res_pts
        except Exception:
            pass
    
    # Phase 3: Simulated Annealing for escape from local optima
    if best_points is not None and time.time() < deadline - 10:
        try:
            sa_points, sa_score = _simulated_annealing(
                best_points, n_iter=3000, t0=0.04, t_end=0.001, seed=42
            )
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_points
        except Exception:
            pass
    
    # Phase 4: Final polishing with L-BFGS-B on barycentric coords
    if best_points is not None and time.time() < deadline - 5:
        try:
            best_bary = _cart_to_bary(best_points)
            res_bary = _local_optimize_lbfgs(best_bary.flatten(), maxiter=300, beta=100.0)
            res_pts = _bary_to_cart(_project_to_bary_simplex(res_bary.reshape(N, 2)))
            res_pts = _clip_to_triangle(res_pts)
            score = _min_triangle_area_fast(res_pts)
            if score > best_score:
                best_score = score
                best_points = res_pts
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        for pts in inits:
            pts = _clip_to_triangle(pts)
            score = _min_triangle_area_fast(pts)
            if score > best_score:
                best_score = score
                best_points = pts
                
    if best_points is None:
        best_points = _generate_fps_initialization(n_points=N, seed=0)
        best_points = _clip_to_triangle(best_points)
        
    return best_points
