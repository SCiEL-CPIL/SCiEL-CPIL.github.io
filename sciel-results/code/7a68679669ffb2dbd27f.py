import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations, product

N = 14

# Precompute triple indices
_TRIPLE_INDICES = np.array(list(combinations(range(N), 3)), dtype=int)
_N_TRIPLES = _TRIPLE_INDICES.shape[0]

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _triangle_areas(points):
    """Compute all triangle areas (unnormalized)."""
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
        
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _soft_min(areas, beta=20.0):
    """
    Smooth approximation of min using log-sum-exp.
    Returns a differentiable lower bound on the min.
    """
    # For numerical stability
    m = np.min(areas)
    if m < 1e-15:
        return m
    # soft min ≈ m - (1/beta) * log(sum(exp(-beta * (areas - m))))
    # But for optimization we want to maximize min, so minimize negative soft min
    # soft_min = - (1/beta) * log(sum(exp(-beta * areas)))
    # For stability: let a = areas - max(areas)
    a = areas - np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * a)))
    soft_min_val = - (1.0 / beta) * log_sum + np.max(areas)
    return soft_min_val

def _soft_score(points, beta=20.0):
    """
    Differentiable soft score for optimization.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    
    areas = _triangle_areas(points)
    sm = _soft_min(areas, beta)
    return sm / hull_area

def _neg_soft_score(points, beta=20.0):
    """Objective for L-BFGS-B: negative soft score."""
    return -_soft_score(points, beta)

def _neg_soft_score_grad(points, beta=20.0):
    """
    Numerical gradient of negative soft score.
    Using central differences for accuracy.
    """
    n = points.size
    grad = np.zeros(n)
    h = 1e-6
    
    f0 = _neg_soft_score(points, beta)
    
    for i in range(n):
        orig = points[i // 2, i % 2]
        points[i // 2, i % 2] = orig + h
        fp = _neg_soft_score(points, beta)
        points[i // 2, i % 2] = orig - h
        fm = _neg_soft_score(points, beta)
        points[i // 2, i % 2] = orig
        grad[i] = (fp - fm) / (2 * h)
    
    return grad

def _generate_ring_config(counts, radii, angles_offset):
    """
    Generate points from multiple rings.
    counts: list of point counts per ring, e.g., [7, 7]
    radii: list of radii per ring, e.g., [1.0, 0.5]
    angles_offset: list of angle offsets per ring
    """
    pts = []
    for c, r, off in zip(counts, radii, angles_offset):
        if c == 0:
            continue
        angles = off + np.linspace(0, 2*np.pi, c, endpoint=False)
        ring_pts = r * np.column_stack([np.cos(angles), np.sin(angles)])
        pts.append(ring_pts)
    if not pts:
        return np.zeros((N, 2))
    return np.vstack(pts)

def _coordinate_polish(points, max_iters=200, step_size=0.01):
    """
    Coordinate-level local search to escape local minima.
    Tries small perturbations in each coordinate direction.
    """
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            for coord in range(2):
                for delta in [step_size, -step_size, step_size*0.5, -step_size*0.5]:
                    trial = best_pts.copy()
                    trial[i, coord] += delta
                    s = _score(trial)
                    if s > best_s + 1e-10:
                        best_pts = trial
                        best_s = s
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Try larger steps
            step_size *= 0.5
            if step_size < 1e-4:
                break
    
    return best_pts, best_s

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    best_score = 0
    best_points = None
    candidates = []

    # 1. Regular 14-gon
    angles_reg = np.linspace(0, 2*np.pi, N, endpoint=False)
    init_reg = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    candidates.append(init_reg)

    # 2. Grid search for 2-ring (7+7)
    best_ring2_score = 0
    best_ring2_pts = None
    r_inner_vals = np.linspace(0.2, 0.8, 10)
    theta_vals = np.linspace(0, np.pi/7, 10, endpoint=False)
    
    for r_in in r_inner_vals:
        for theta in theta_vals:
            if time.time() > deadline - 50:
                break
            counts = [7, 7]
            radii = [1.0, r_in]
            offsets = [0.0, theta]
            pts = _generate_ring_config(counts, radii, offsets)
            s = _score(pts)
            if s > best_ring2_score:
                best_ring2_score = s
                best_ring2_pts = pts.copy()
        
        if time.time() > deadline - 50:
            break
            
    if best_ring2_pts is not None:
        candidates.append(best_ring2_pts)

    # 3. Grid search for 3-ring (6+2+6)
    best_ring3_score = 0
    best_ring3_pts = None
    r_mid_vals = np.linspace(0.4, 0.8, 6)
    r_in_vals = np.linspace(0.1, 0.5, 6)
    theta_mid_vals = np.linspace(0, np.pi, 5, endpoint=False)
    theta_in_vals = np.linspace(0, np.pi/6, 5, endpoint=False)
    
    for r_mid in r_mid_vals:
        for r_in in r_in_vals:
            for th_mid in theta_mid_vals:
                for th_in in theta_in_vals:
                    if time.time() > deadline - 35:
                        break
                    counts = [6, 2, 6]
                    radii = [1.0, r_mid, r_in]
                    offsets = [0.0, th_mid, th_in]
                    pts = _generate_ring_config(counts, radii, offsets)
                    s = _score(pts)
                    if s > best_ring3_score:
                        best_ring3_score = s
                        best_ring3_pts = pts.copy()
                    if time.time() > deadline - 35:
                        break
                if time.time() > deadline - 35:
                    break
            if time.time() > deadline - 35:
                break
        if time.time() > deadline - 35:
            break
            
    if best_ring3_pts is not None:
        candidates.append(best_ring3_pts)

    # 4. Add random candidates to break symmetry
    np.random.seed(42)
    for _ in range(8):
        if time.time() > deadline - 20:
            break
        pts = np.random.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / np.maximum(norms, 1e-6)[:, np.newaxis]
        candidates.append(pts)

    # 5. Score and sort candidates
    scored_candidates = []
    for pts in candidates:
        s = _score(pts)
        scored_candidates.append((s, pts))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Deduplicate
    unique_candidates = []
    for s, pts in scored_candidates:
        if not unique_candidates:
            unique_candidates.append((s, pts))
        else:
            too_similar = False
            for _, existing_pts in unique_candidates:
                dist = np.linalg.norm(pts - existing_pts)
                if dist < 0.15:
                    too_similar = True
                    break
            if not too_similar:
                unique_candidates.append((s, pts))
        
        if len(unique_candidates) >= 6:
            break

    # 6. Optimization using L-BFGS-B with soft-min objective
    # Try multiple beta values for different levels of smoothness
    betas = [10.0, 20.0, 40.0]
    
    for s_init, pts_init in unique_candidates:
        if time.time() > deadline - 15:
            break
            
        for beta in betas:
            if time.time() > deadline - 15:
                break
                
            try:
                # L-BFGS-B with numerical gradient
                res = minimize(
                    lambda x: _neg_soft_score(x, beta),
                    pts_init,
                    method='L-BFGS-B',
                    jac=lambda x: _neg_soft_score_grad(x, beta),
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 5000, 'disp': False}
                )
                
                s_opt = _score(res.x)
                if s_opt > best_score:
                    best_score = s_opt
                    best_points = res.x.copy()
            except Exception:
                pass

    # 7. Coordinate polish on the best result
    if best_points is not None:
        if time.time() < deadline - 5:
            polished_pts, polished_s = _coordinate_polish(best_points, max_iters=100, step_size=0.02)
            if polished_s > best_score:
                best_score = polished_s
                best_points = polished_pts
    else:
        # Fallback: use best initial candidate
        if scored_candidates:
            best_score = scored_candidates[0][0]
            best_points = scored_candidates[0][1]
            if time.time() < deadline - 5:
                polished_pts, polished_s = _coordinate_polish(best_points, max_iters=100, step_size=0.02)
                if polished_s > best_score:
                    best_score = polished_s
                    best_points = polished_pts
        else:
            best_points = init_reg
            best_score = _score(init_reg)
    
    return best_points
