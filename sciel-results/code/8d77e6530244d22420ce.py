import numpy as np
import time
from scipy.optimize import minimize

def _vertex_score(distances_sorted):
    d = distances_sorted
    n = len(d)
    if n != 19:
        return 0.0
    if np.any(d < 0.05):
        return 0.0
    diffs = d[1:] - d[:-1]
    min_diff = np.min(diffs)
    # Find first index where the minimum gap occurs
    k = int(np.argmax(diffs == min_diff))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        four_d = d[0:4]
    elif k == 17:
        four_d = d[15:19]
    else:
        four_d = d[k-1:k+3]
    if np.any(four_d <= 0):
        return 0.0
    ratios = np.maximum(four_d / D, D / four_d)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
    return float(score)

def _is_valid_convex(p):
    n = len(p)
    if not np.all(np.isfinite(p)):
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-10:
                return False
    # Cross products all same sign and nonzero
    cross_products = np.zeros(n)
    for i in range(n):
        e1 = p[(i+1) % n] - p[i]
        e2 = p[(i+2) % n] - p[(i+1) % n]
        cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
    if np.any(np.abs(cross_products) < 1e-14):
        return False
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return False
    # Total turning
    total_turn = 0.0
    for i in range(n):
        v1 = p[(i+1) % n] - p[i]
        v2 = p[(i+2) % n] - p[(i+1) % n]
        ang1 = np.arctan2(v1[1], v1[0])
        ang2 = np.arctan2(v2[1], v2[0])
        turn = ang2 - ang1
        while turn > np.pi:
            turn -= 2 * np.pi
        while turn < -np.pi:
            turn += 2 * np.pi
        total_turn += turn
    if not np.isclose(abs(total_turn), 2 * np.pi, atol=1e-6):
        return False
    return True

def _compute_min_vertex_score(polygon):
    n = len(polygon)
    if n != 20:
        return 0.0
    if not np.all(np.isfinite(polygon)):
        return 0.0
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist

    if not _is_valid_convex(p):
        return 0.0

    # Vectorized pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))

    min_score = 1.0
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        score = _vertex_score(d_sorted)
        min_score = min(min_score, score)
    return min_score

def _soft_min_vertex_score(polygon, beta=50.0):
    """Smooth surrogate: soft-min of vertex scores using log-sum-exp."""
    n = len(polygon)
    if n != 20:
        return 0.0
    if not np.all(np.isfinite(polygon)):
        return 0.0
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist

    if not _is_valid_convex(p):
        return 0.0

    # Vectorized pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))

    scores = np.zeros(n)
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        scores[i] = _vertex_score(d_sorted)
    
    # Soft min: -log(sum(exp(-beta * s_i))) / beta
    # For large beta, this approaches min(s_i)
    # Clip to avoid overflow
    scores_clipped = np.clip(scores, 1e-10, 1.0)
    neg_beta_scores = -beta * scores_clipped
    max_neg = np.max(neg_beta_scores)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_beta_scores - max_neg)))
    soft_min = -log_sum / beta
    return float(soft_min)

def _regular_polygon():
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0

    # Start with regular polygon
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)

    if time.time() > deadline - 30:
        return best_poly

    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def _radial_poly_from_params(params):
        # params: array of 9 Fourier coefficients
        radii = np.ones(n)
        for k in range(1, 10):
            radii += params[k-1] * np.cos(k * base_angles)
        # Ensure radii are positive
        radii = np.maximum(radii, 0.1)
        x = radii * np.cos(base_angles)
        y = radii * np.sin(base_angles)
        return np.stack([x, y], axis=1)

    def _objective_soft(params):
        poly = _radial_poly_from_params(params)
        return -_soft_min_vertex_score(poly, beta=30.0)

    def _objective_exact(params):
        poly = _radial_poly_from_params(params)
        return -_compute_min_vertex_score(poly)

    # Generate diverse initial conditions
    initial_params_list = []
    # Zero (regular polygon)
    initial_params_list.append(np.zeros(9))
    # Small random perturbations
    for _ in range(15):
        initial_params_list.append(np.random.uniform(-0.08, 0.08, 9))
    # Low-frequency dominant patterns
    for m in range(1, 8):
        for amp in [0.02, 0.04, 0.06, 0.08]:
            for sign in [1, -1]:
                p = np.zeros(9)
                p[m-1] = sign * amp
                initial_params_list.append(p.copy())
    # Mixed frequency patterns
    for _ in range(10):
        p = np.zeros(9)
        for m in range(1, 5):
            p[m-1] = np.random.uniform(-0.05, 0.05)
        initial_params_list.append(p)

    best_radial_poly = best_poly.copy()
    best_radial_score = best_score
    
    # Phase 1: Optimize with soft-min surrogate
    for ip in initial_params_list:
        if time.time() > deadline - 120.0:
            break
        try:
            result = minimize(
                _objective_soft,
                ip,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            # Evaluate with exact score
            exact_score = _compute_min_vertex_score(_radial_poly_from_params(result.x))
            if exact_score > best_radial_score + 1e-9:
                best_radial_score = exact_score
                best_radial_poly = _radial_poly_from_params(result.x)
        except Exception:
            continue
    
    if best_radial_score > best_score:
        best_score = best_radial_score
        best_poly = best_radial_poly

    if time.time() > deadline - 90.0:
        return best_poly

    # Phase 2: Refine with exact objective using Nelder-Mead
    x0 = best_poly.flatten()
    try:
        result = minimize(
            _objective_exact,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        score = -result.fun
        if score > best_score + 1e-9:
            best_score = score
            best_poly = result.x.reshape(20, 2)
    except Exception:
        pass

    if time.time() > deadline - 60.0:
        return best_poly

    # Phase 3: Full coordinate optimization with soft-min
    def _objective_full_soft(flat_params):
        poly = flat_params.reshape(20, 2)
        return -_soft_min_vertex_score(poly, beta=30.0)

    def _objective_full_exact(flat_params):
        poly = flat_params.reshape(20, 2)
        return -_compute_min_vertex_score(poly)

    x0 = best_poly.flatten()
    try:
        result = minimize(
            _objective_full_soft,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 150, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
        )
        score = _compute_min_vertex_score(result.x.reshape(20, 2))
        if score > best_score + 1e-9:
            best_score = score
            best_poly = result.x.reshape(20, 2)
    except Exception:
        pass

    if time.time() > deadline - 30.0:
        return best_poly

    # Phase 4: Final high-precision polish
    x0 = best_poly.flatten()
    try:
        result = minimize(
            _objective_full_exact,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        score = -result.fun
        if score > best_score + 1e-9:
            best_score = score
            best_poly = result.x.reshape(20, 2)
    except Exception:
        pass

    # Phase 5: Coordinate descent refinement
    current_poly = best_poly.copy()
    current_score = best_score
    step_size = 0.005
    no_improve = 0
    
    while time.time() < deadline - 10.0 and no_improve < 40:
        improved = False
        for i in range(20):
            if time.time() > deadline - 10.0:
                break
            old_v = current_poly[i].copy()
            best_step = None
            best_local_score = current_score
            for dx in [-step_size, 0.0, step_size]:
                for dy in [-step_size, 0.0, step_size]:
                    if dx == 0.0 and dy == 0.0:
                        continue
                    new_poly = current_poly.copy()
                    new_poly[i] = old_v + np.array([dx, dy])
                    s = _compute_min_vertex_score(new_poly)
                    if s > best_local_score + 1e-12:
                        best_local_score = s
                        best_step = new_poly
            if best_step is not None:
                current_poly = best_step
                current_score = best_local_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_poly = current_poly.copy()
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            step_size *= 0.5
            if step_size < 1e-7:
                break

    return best_poly
