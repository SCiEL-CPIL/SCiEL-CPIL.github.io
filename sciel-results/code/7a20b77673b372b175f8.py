import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# Precompute triple indices for vectorized area computation
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

# Precompute which triples involve each point for coordinate descent
POINT_TRIPLE_MAP = [[] for _ in range(N)]
for idx, (i, j, k) in enumerate(TRIPLE_INDICES):
    POINT_TRIPLE_MAP[i].append(idx)
    POINT_TRIPLE_MAP[j].append(idx)
    POINT_TRIPLE_MAP[k].append(idx)
POINT_TRIPLE_ARRAYS = [np.array(pt) for pt in POINT_TRIPLE_MAP]

def _compute_areas(points):
    """Compute all triangle areas using cross-product formula."""
    p = points
    C = p[:, 0:1] * p[:, 1:2].T - p[:, 1:2] * p[:, 0:1].T
    i_idx = TRIPLE_INDICES[:, 0]
    j_idx = TRIPLE_INDICES[:, 1]
    k_idx = TRIPLE_INDICES[:, 2]
    vals = C[i_idx, j_idx] + C[j_idx, k_idx] - C[i_idx, k_idx]
    return 0.5 * np.abs(vals)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _calculate_score(points):
    """Exact score: min triangle area / hull area."""
    if points.shape != (N, 2):
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0
    areas = _compute_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_score(points, temperature=0.01):
    """Smooth surrogate using log-sum-exp for the minimum triangle area."""
    if points.shape != (N, 2):
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0
    areas = _compute_areas(points)
    max_a = np.max(areas)
    if max_a < 1e-12:
        return 0.0
    shifted = areas - max_a
    log_sum = np.log(np.sum(np.exp(shifted / temperature)))
    soft_min_val = max_a - temperature * log_sum
    return soft_min_val / hull_area

def _epigraph_score(points, t, temperature=0.005):
    """
    Epigraph-style smooth objective: maximize t subject to all triangle areas >= t.
    We use a smooth penalty: score = t - lambda * sum(max(0, t - area_i)^2) / hull_area
    This creates a smooth landscape that guides toward the true minimum.
    """
    if points.shape != (N, 2):
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0
    areas = _compute_areas(points)
    
    # Soft violation: how much each area falls below t
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    
    # Smooth score: t minus penalty, normalized
    score = t - penalty / hull_area
    return score

def _generate_two_ring(n_out, n_in, r_in, phi_rel):
    """Generate points on two concentric rings."""
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    if n_in > 0:
        angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + phi_rel
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        return np.vstack((pts_out, pts_in))
    else:
        return pts_out

def _generate_fourier(a1, b1, a2, b2, a3, b3):
    """Generate points using Fourier radius parameterization around a circle."""
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = 1.0 + a1*np.cos(angles) + b1*np.sin(angles) + \
             a2*np.cos(2*angles) + b2*np.sin(2*angles) + \
             a3*np.cos(3*angles) + b3*np.sin(3*angles)
    r = np.maximum(r, 0.3)
    pts = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
    return pts

def _coordinate_descent_polish(points, time_limit=3.0):
    """
    Iteratively adjust each point to improve the minimum triangle area.
    Focus on points involved in the worst (smallest area) triangles.
    """
    start_time = time.time()
    pts = points.copy()
    score = _calculate_score(pts)
    
    for iteration in range(50):
        if time.time() - start_time > time_limit:
            break
        
        areas = _compute_areas(pts)
        min_area = np.min(areas)
        current_score = min_area / _hull_area(pts)
        
        # Find the worst triangles (bottom 10%)
        n_worst = max(3, len(areas) // 10)
        worst_indices = np.argsort(areas)[:n_worst]
        
        # Get unique points involved in worst triangles
        worst_points = set()
        for idx in worst_indices:
            i, j, k = TRIPLE_INDICES[idx]
            worst_points.add(i)
            worst_points.add(j)
            worst_points.add(k)
        
        # Try to improve each worst point
        for pt_idx in sorted(worst_points):
            if time.time() - start_time > time_limit:
                break
            
            original = pts[pt_idx].copy()
            best_local = original
            best_local_score = current_score
            
            # Try small perturbations in multiple directions
            step_sizes = [0.01, 0.005, 0.002]
            directions = [
                np.array([1, 0]), np.array([-1, 0]),
                np.array([0, 1]), np.array([0, -1]),
                np.array([1, 1])/np.sqrt(2), np.array([1, -1])/np.sqrt(2),
                np.array([-1, 1])/np.sqrt(2), np.array([-1, -1])/np.sqrt(2),
            ]
            
            for step in step_sizes:
                for d in directions:
                    new_pt = pts.copy()
                    new_pt[pt_idx] = original + step * d
                    new_score = _calculate_score(new_pt)
                    if new_score > best_local_score:
                        best_local_score = new_score
                        best_local = new_pt[pt_idx].copy()
            
            pts[pt_idx] = best_local
        
        new_score = _calculate_score(pts)
        if new_score > score:
            score = new_score
        else:
            # No improvement in this iteration, try smaller steps
            if step_sizes[0] < 0.001:
                break
    
    return pts, _calculate_score(pts)

def _polish_points(points, time_limit=5.0):
    """
    Multi-phase polish: L-BFGS-B with soft surrogate, then coordinate descent,
    then final exact local search.
    """
    start_time = time.time()
    best_pts = points.copy()
    best_score = _calculate_score(points)
    
    # Phase 1: L-BFGS-B with soft-min surrogate (temperature 0.05)
    temp1 = 0.05
    def neg_smooth_score1(x):
        p = x.reshape(N, 2)
        return -_soft_min_score(p, temp1)
    
    try:
        res1 = minimize(neg_smooth_score1, points.flatten(), method='L-BFGS-B',
                        options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 40})
        new_pts1 = res1.x.reshape(N, 2)
        score1 = _calculate_score(new_pts1)
        if score1 > best_score:
            best_pts = new_pts1
            best_score = score1
    except Exception:
        pass

    # Phase 2: L-BFGS-B with lower temperature (0.01)
    if time.time() - start_time < time_limit * 0.4:
        temp2 = 0.01
        def neg_smooth_score2(x):
            p = x.reshape(N, 2)
            return -_soft_min_score(p, temp2)
        
        try:
            res2 = minimize(neg_smooth_score2, best_pts.flatten(), method='L-BFGS-B',
                            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 40})
            new_pts2 = res2.x.reshape(N, 2)
            score2 = _calculate_score(new_pts2)
            if score2 > best_score:
                best_pts = new_pts2
                best_score = score2
        except Exception:
            pass

    # Phase 3: Coordinate descent targeting worst triangles
    if time.time() - start_time < time_limit * 0.7:
        cd_time = min(2.0, time_limit * 0.2)
        try:
            cd_pts, cd_score = _coordinate_descent_polish(best_pts, cd_time)
            if cd_score > best_score:
                best_pts = cd_pts
                best_score = cd_score
        except Exception:
            pass

    # Phase 4: Final exact local polish with Nelder-Mead
    if time.time() - start_time < time_limit * 0.9:
        def neg_exact_score(x):
            p = x.reshape(N, 2)
            return -_calculate_score(p)
        
        try:
            res3 = minimize(neg_exact_score, best_pts.flatten(), method='Nelder-Mead',
                            options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 300, 'disp': False})
            new_pts3 = res3.x.reshape(N, 2)
            score3 = _calculate_score(new_pts3)
            if score3 > best_score:
                best_pts = new_pts3
                best_score = score3
        except Exception:
            pass

    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_score = 0
    best_points = None
    
    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Strategy 1: Two-ring configurations
    splits = [
        (14, 0),
        (13, 1),
        (12, 2),
        (11, 3),
        (10, 4),
        (9, 5),
        (8, 6),
        (7, 7),
    ]
    
    for n_out, n_in in splits:
        if time.time() > deadline - 20.0:
            break
        
        if n_in == 0:
            angles = np.linspace(0, 2*np.pi, n_out, endpoint=False)
            pts = np.column_stack((np.cos(angles), np.sin(angles)))
        else:
            def ring_objective(params):
                r_in, phi_rel = params
                if r_in < 0.0 or r_in > 1.5:
                    return 1e6
                pts = _generate_two_ring(n_out, n_in, r_in, phi_rel)
                return -_calculate_score(pts)
            
            best_r = 0.5
            best_phi = 0.0
            best_obj = 1e9
            
            r_starts = [0.2, 0.4, 0.6, 0.8, 1.0]
            phi_starts = [0.0, np.pi/n_in, 2*np.pi/n_in]
            
            for r0 in r_starts:
                for phi0 in phi_starts:
                    if time.time() > deadline - 15.0:
                        break
                    try:
                        res = minimize(ring_objective, [r0, phi0], method='Nelder-Mead',
                                       options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 80})
                        if res.fun < best_obj:
                            best_obj = res.fun
                            best_r = res.x[0]
                            best_phi = res.x[1]
                    except Exception:
                        pass
            
            pts = _generate_two_ring(n_out, n_in, best_r, best_phi)
        
        remaining = deadline - time.time()
        if remaining > 8.0:
            polish_time = min(6.0, remaining * 0.35)
            polished_pts, polished_score = _polish_points(pts, time_limit=polish_time)
            try_update(polished_score, polished_pts)
    
    # Strategy 2: Fourier parameterization for diverse shapes
    if time.time() < deadline - 25.0:
        def fourier_objective(params):
            a1, b1, a2, b2, a3, b3 = params
            if any(abs(p) > 0.25 for p in params):
                return 1e6
            pts = _generate_fourier(a1, b1, a2, b2, a3, b3)
            return -_calculate_score(pts)
        
        np.random.seed(42)
        for trial in range(8):
            if time.time() > deadline - 15.0:
                break
            x0 = np.random.uniform(-0.12, 0.12, 6)
            try:
                res = minimize(fourier_objective, x0, method='Nelder-Mead',
                               options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 150})
                if -res.fun > 0:
                    pts = _generate_fourier(*res.x)
                    remaining = deadline - time.time()
                    if remaining > 10.0:
                        polish_time = min(5.0, remaining * 0.3)
                        polished_pts, polished_score = _polish_points(pts, time_limit=polish_time)
                        try_update(polished_score, polished_pts)
            except Exception:
                pass
    
    # Final intensive polish on the best points found so far
    if best_points is not None and time.time() < deadline - 10.0:
        remaining = deadline - time.time() - 3.0
        if remaining > 5.0:
            final_pts, final_score = _polish_points(best_points, time_limit=remaining)
            try_update(final_score, final_pts)
    
    # Fallback if nothing worked
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _calculate_score(best_points)

    return best_points
