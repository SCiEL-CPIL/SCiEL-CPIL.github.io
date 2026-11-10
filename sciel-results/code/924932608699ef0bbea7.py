# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

# Precompute triangle indices for 13 points
_N = 13
_I_IDX, _J_IDX, _K_IDX = [], [], []
for _i in range(_N):
    for _j in range(_i + 1, _N):
        for _k in range(_j + 1, _N):
            _I_IDX.append(_i)
            _J_IDX.append(_j)
            _K_IDX.append(_k)
_I_IDX = np.array(_I_IDX)
_J_IDX = np.array(_J_IDX)
_K_IDX = np.array(_K_IDX)


def _all_triangle_areas(pts):
    """Compute all 286 triangle areas vectorized using precomputed indices."""
    x1, y1 = pts[_I_IDX, 0], pts[_I_IDX, 1]
    x2, y2 = pts[_J_IDX, 0], pts[_J_IDX, 1]
    x3, y3 = pts[_K_IDX, 0], pts[_K_IDX, 1]
    areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
    return areas


def _soft_min_triangle_area(pts, beta=50.0):
    """
    Smooth approximation of min triangle area using log-sum-exp.
    soft_min = -logsumexp(-areas / beta) * beta
    As beta -> 0, this approaches the true min.
    """
    areas = _all_triangle_areas(pts)
    # Clip to avoid numerical issues
    areas_clipped = np.maximum(areas, 1e-15)
    # Log-sum-exp trick for stability
    max_a = np.max(areas_clipped)
    log_sum = np.log(np.sum(np.exp(-areas_clipped / beta))) + (-max_a / beta)
    soft_min = -beta * log_sum + max_a
    # Actually: soft_min = beta * logsumexp(areas/beta) - ... 
    # Let me redo: soft_min = -beta * log(sum(exp(-a_i/beta)))
    # = -beta * (log_sum_exp(-a_i/beta))
    # log_sum_exp(z) = max(z) + log(sum(exp(z - max(z))))
    z = -areas_clipped / beta
    z_max = np.max(z)
    lse = z_max + np.log(np.sum(np.exp(z - z_max)))
    soft_min_val = -beta * lse
    return soft_min_val


def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _soft_score(pts, beta=50.0):
    """
    Smooth surrogate score: soft_min_area / hull_area
    """
    if len(pts) < 3:
        return 0.0
    sm = _soft_min_triangle_area(pts, beta)
    ah = _hull_area(pts)
    if ah <= 0 or not np.isfinite(sm) or not np.isfinite(ah):
        return 0.0
    return sm / ah


def _exact_score(pts):
    """Compute A_min / A_hull exactly."""
    if len(pts) < 3:
        return 0.0
    areas = _all_triangle_areas(pts)
    a_min = np.min(areas)
    a_hull = _hull_area(pts)
    if a_hull <= 0 or not np.isfinite(a_min) or not np.isfinite(a_hull):
        return 0.0
    return a_min / a_hull


def _objective_neg(x, pts_array):
    """Negative score for minimization."""
    score = _exact_score(pts_array)
    if score <= 0:
        return 1e10
    return -score


def _objective_soft_neg(x, beta=50.0):
    """Negative soft score for smooth optimization."""
    pts = x.reshape(13, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    score = _soft_score(pts, beta)
    if score <= 0:
        return 1e10
    return -score


def _make_d3_symmetric(pts_3):
    """
    Take 3 base points and generate 13 points using D3 symmetry.
    D3 has 6 elements. 13 = 6*2 + 1.
    We place 1 point at center, and 6 points in two orbits of 3.
    Actually, let's use: 1 center + 12 in two hexagonal rings (6 each)
    or 1 center + 3 + 9 (triangle + 3x3)
    
    Better: use 1 center point, and arrange remaining 12 in a pattern
    that respects D3 symmetry.
    
    Simplest: 1 center, 3 points forming equilateral triangle (orbit 1),
    and 9 points forming a larger equilateral triangle pattern (orbit 2 with 3 sub-orbits)
    
    Let's just use: 1 center, 6 in inner hexagon, 6 in outer hexagon = 13
    That's D6 symmetry, which is a superset of D3.
    """
    center = np.array([0.0, 0.0])
    # Inner hexagon: 6 points
    r_inner = pts_3[0, 0]
    angle_offset_inner = pts_3[0, 1]
    inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + angle_offset_inner
    inner_pts = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    
    # Outer hexagon: 6 points
    r_outer = pts_3[1, 0]
    angle_offset_outer = pts_3[1, 1]
    outer_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + angle_offset_outer
    outer_pts = np.column_stack([r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles)])
    
    # 1 center + 6 inner + 6 outer = 13
    all_pts = np.vstack([center.reshape(1, 2), inner_pts, outer_pts])
    return all_pts


def _make_concentric_rings(r1, r2, angle_off1, angle_off2):
    """Create points in two concentric hexagonal rings + center."""
    center = np.array([0.0, 0.0])
    inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + angle_off1
    inner_pts = np.column_stack([r1 * np.cos(inner_angles), r1 * np.sin(inner_angles)])
    outer_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + angle_off2
    outer_pts = np.column_stack([r2 * np.cos(outer_angles), r2 * np.sin(outer_angles)])
    return np.vstack([center.reshape(1, 2), inner_pts, outer_pts])


def _make_perturbed_symmetric(base_pts, rng, noise_scale):
    """Perturb a symmetric configuration."""
    return base_pts + rng.normal(0, noise_scale, base_pts.shape)


def heilbronn_convex13() -> np.ndarray:
    """
    Find 13 points maximizing A_min / A_hull.
    Uses smooth surrogate for optimization, then exact polish.
    """
    start_time = time.time()
    time_limit = 590  # Safety margin from 630s
    
    def time_check(remaining=30):
        return time.time() - start_time < time_limit - remaining
    
    # Best solution tracking
    best_pts = None
    best_score = 0.0
    
    def update_best(pts):
        nonlocal best_pts, best_score
        if pts is None or not np.all(np.isfinite(pts)) or len(pts) != 13:
            return
        score = _exact_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # === Phase 1: Generate diverse initial configurations ===
    initial_configs = []
    
    # 1. Concentric hexagons with various radii and angles
    for r1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for r2 in [0.9, 1.0, 1.1, 1.2]:
            for a1 in [0, np.pi/6, np.pi/3]:
                for a2 in [0, np.pi/6, np.pi/3]:
                    if r1 < r2:
                        pts = _make_concentric_rings(r1, r2, a1, a2)
                        initial_configs.append(pts)
    
    # 2. Regular 13-gon
    angles_13 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    initial_configs.append(np.column_stack([np.cos(angles_13), np.sin(angles_13)]))
    
    # 3. Fibonacci spiral
    golden_angle = np.pi * (3 - np.sqrt(5))
    fib_pts = []
    for i in range(13):
        r = np.sqrt(i + 1) * 0.3
        theta = i * golden_angle
        fib_pts.append([r * np.cos(theta), r * np.sin(theta)])
    initial_configs.append(np.array(fib_pts))
    
    # 4. Random perturbations of symmetric configs
    rng = np.random.default_rng(42)
    for trial in range(30):
        r1 = rng.uniform(0.25, 0.75)
        r2 = rng.uniform(0.85, 1.35)
        a1 = rng.uniform(0, 2 * np.pi)
        a2 = rng.uniform(0, 2 * np.pi)
        base = _make_concentric_rings(r1, r2, a1, a2)
        perturbed = _make_perturbed_symmetric(base, rng, 0.15)
        initial_configs.append(perturbed)
    
    # === Phase 2: Evaluate all initial configs ===
    for pts in initial_configs:
        if not time_check(100):
            break
        update_best(pts)
    
    # === Phase 3: Score-based ranking and local optimization ===
    scored_configs = []
    for pts in initial_configs:
        score = _exact_score(pts)
        scored_configs.append((score, pts))
    scored_configs.sort(key=lambda x: -x[0])
    
    top_configs = [cfg for _, cfg in scored_configs[:12]]
    
    for init_pts in top_configs:
        if not time_check(80):
            break
        
        x0 = init_pts.flatten()
        
        # Use smooth surrogate with decreasing beta for better convergence
        for beta in [100.0, 50.0, 20.0, 10.0]:
            if not time_check(60):
                break
            
            def obj_soft(x, b=beta):
                pts = x.reshape(13, 2)
                if not np.all(np.isfinite(pts)):
                    return 1e10
                score = _soft_score(pts, b)
                if score <= 0:
                    return 1e10
                return -score
            
            try:
                result = minimize(
                    obj_soft,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 3000,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                if np.all(np.isfinite(result.x)):
                    candidate = result.x.reshape(13, 2)
                    cand_score = _exact_score(candidate)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = candidate.copy()
                        x0 = candidate.flatten()
            except Exception:
                pass
        
        # Exact Nelder-Mead polish
        if not time_check(50):
            break
        
        def obj_exact(x):
            pts = x.reshape(13, 2)
            if not np.all(np.isfinite(pts)):
                return 1e10
            score = _exact_score(pts)
            if score <= 0:
                return 1e10
            return -score
        
        try:
            result = minimize(
                obj_exact,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-13,
                    'fatol': 1e-15,
                    'adaptive': True
                }
            )
            if np.all(np.isfinite(result.x)):
                candidate = result.x.reshape(13, 2)
                cand_score = _exact_score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = candidate.copy()
        except Exception:
            pass
    
    # === Phase 4: Multi-start refinement from best solution ===
    if time_check(60) and best_pts is not None:
        x0 = best_pts.flatten()
        
        # Multiple perturbations and re-optimization
        for trial in range(8):
            if not time_check(40):
                break
            
            rng_local = np.random.default_rng(200 + trial)
            for scale in [0.02, 0.05, 0.1]:
                if not time_check(30):
                    break
                x_pert = x0 + rng_local.normal(0, scale, 26)
                
                # Smooth optimization
                def obj_soft2(x):
                    pts = x.reshape(13, 2)
                    if not np.all(np.isfinite(pts)):
                        return 1e10
                    score = _soft_score(pts, 30.0)
                    if score <= 0:
                        return 1e10
                    return -score
                
                try:
                    res = minimize(
                        obj_soft2,
                        x_pert,
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                    )
                    if np.all(np.isfinite(res.x)):
                        cand = res.x.reshape(13, 2)
                        cs = _exact_score(cand)
                        if cs > best_score:
                            best_score = cs
                            best_pts = cand.copy()
                            x0 = cand.flatten()
                except Exception:
                    pass
    
    # === Phase 5: Final high-precision polish ===
    if time_check(30) and best_pts is not None and best_score > 0:
        x0 = best_pts.flatten()
        
        def obj_final(x):
            pts = x.reshape(13, 2)
            if not np.all(np.isfinite(pts)):
                return 1e10
            score = _exact_score(pts)
            if score <= 0:
                return 1e10
            return -score
        
        try:
            result = minimize(
                obj_final,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'xatol': 1e-14,
                    'fatol': 1e-16,
                    'adaptive': True
                }
            )
            if np.all(np.isfinite(result.x)):
                candidate = result.x.reshape(13, 2)
                cand_score = _exact_score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = candidate.copy()
        except Exception:
            pass
        
        # Also try Powell for final polish
        if time_check(15):
            try:
                result_p = minimize(
                    obj_final,
                    best_pts.flatten(),
                    method='Powell',
                    options={'maxiter': 2000, 'xtol': 1e-14, 'ftol': 1e-16}
                )
                if np.all(np.isfinite(result_p.x)):
                    candidate = result_p.x.reshape(13, 2)
                    cand_score = _exact_score(candidate)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = candidate.copy()
            except Exception:
                pass
    
    # === Final validation and return ===
    if best_pts is None or not np.all(np.isfinite(best_pts)) or len(best_pts) != 13:
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts


# EVOLVE-BLOCK-END
