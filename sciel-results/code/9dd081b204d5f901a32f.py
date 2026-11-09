# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _vertex_score(distances_sorted):
    d = distances_sorted
    n = len(d)
    if n != 19:
        return 0.0
    if np.any(d < 0.05):
        return 0.0
    diffs = d[1:] - d[:-1]
    min_diff = np.min(diffs)
    k = int(np.argmin(diffs))
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
    for i in range(n):
        for j in range(i+1, n):
            if np.allclose(p[i], p[j], atol=1e-10):
                return False
    cross_products = np.array([
        (p[(i+1)%n] - p[i])[0] * (p[(i+2)%n] - p[(i+1)%n])[1] -
        (p[(i+1)%n] - p[i])[1] * (p[(i+2)%n] - p[(i+1)%n])[0]
        for i in range(n)
    ])
    if np.any(np.abs(cross_products) < 1e-12):
        return False
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return False
    total_turn = 0.0
    for i in range(n):
        v1 = p[(i+1) % n] - p[i]
        v2 = p[(i+2) % n] - p[(i+1) % n]
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        turn = angle2 - angle1
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
    """Smooth surrogate for min vertex score using log-sum-exp."""
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
    
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    scores = np.zeros(n)
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        scores[i] = _vertex_score(d_sorted)
    
    # Soft min: -log(sum(exp(-beta * s))) / beta
    # Use stable computation
    min_s = np.min(scores)
    if min_s <= 0:
        return 0.0
    exp_scores = np.exp(-beta * (scores - min_s))
    soft_min = min_s - np.log(np.sum(exp_scores)) / beta
    return float(soft_min)

def _regular_polygon():
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.stack([x, y], axis=1)

def _fourier_polygon(fourier_coeffs):
    """
    Reconstruct polygon from Fourier coefficients.
    Supports harmonics 1, 2, 3, 4.
    coeffs: [a1, b1, a2, b2, a3, b3, a4, b4]
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = np.ones(n)
    for h in range(1, 5):
        idx_a = 2 * (h - 1)
        idx_b = 2 * (h - 1) + 1
        if idx_a < len(fourier_coeffs):
            radii += fourier_coeffs[idx_a] * np.cos(h * base_angles) + fourier_coeffs[idx_b] * np.sin(h * base_angles)
    x = radii * np.cos(base_angles)
    y = radii * np.sin(base_angles)
    return np.stack([x, y], axis=1)

def _angular_polygon(theta_perturbations, radius_perturbations=None):
    """
    Construct polygon from angular perturbations and optional radial perturbations.
    theta_perturbations: 20 angles of perturbation from regular 20-gon positions.
    radius_perturbations: optional 20 radii (default 1.0).
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles = base_angles + theta_perturbations
    
    if radius_perturbations is None:
        radii = np.ones(n)
    else:
        radii = 1.0 + radius_perturbations
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _negative_score_fourier(fourier_coeffs, use_soft=False, beta=50.0):
    poly = _fourier_polygon(fourier_coeffs)
    if use_soft:
        score = _soft_min_vertex_score(poly, beta)
    else:
        score = _compute_min_vertex_score(poly)
    return -score

def _negative_score_angular(params, use_soft=False, beta=50.0):
    """
    params: [theta_0, ..., theta_19, r_0, ..., r_19] (40 total)
    or just [theta_0, ..., theta_19] (20 total) if no radial perturbations.
    """
    n = 20
    if len(params) >= 40:
        theta = params[:20]
        radii = params[20:40]
        poly = _angular_polygon(theta, radii)
    else:
        theta = params[:20]
        poly = _angular_polygon(theta)
    
    if use_soft:
        score = _soft_min_vertex_score(poly, beta)
    else:
        score = _compute_min_vertex_score(poly)
    return -score

def _optimize_fourier_global(deadline):
    """Use differential evolution to find good Fourier coefficients."""
    start_time = time.time()
    remaining = deadline - start_time
    if remaining < 30:
        return _regular_polygon(), 0.0
    
    # Use 4 harmonics (8 coefficients)
    bounds = [(-0.3, 0.3), (-0.3, 0.3),
              (-0.2, 0.2), (-0.2, 0.2),
              (-0.1, 0.1), (-0.1, 0.1),
              (-0.05, 0.05), (-0.05, 0.05)]
    
    # Scale iterations based on available time
    maxiter = max(20, int(remaining / 5))
    popsize = max(15, min(30, int(remaining / 10)))
    
    result = differential_evolution(
        _negative_score_fourier,
        bounds,
        maxiter=maxiter,
        tol=1e-8,
        popsize=popsize,
        seed=42,
        polish=True
    )
    
    best_fourier = result.x
    best_score = -result.fun
    best_poly = _fourier_polygon(best_fourier)
    
    return best_poly, best_score

def _optimize_angular_global(deadline):
    """
    Global search in angular + radial perturbation space.
    params: [theta_0..theta_19, r_0..r_19]
    """
    start_time = time.time()
    remaining = deadline - start_time
    if remaining < 30:
        return _regular_polygon(), 0.0
    
    # Angular perturbations: small to preserve convexity
    # Radial perturbations: moderate
    theta_bounds = [(-0.15, 0.15)] * 20
    radius_bounds = [(-0.2, 0.2)] * 20
    bounds = theta_bounds + radius_bounds
    
    # Scale iterations based on available time
    maxiter = max(15, int(remaining / 8))
    popsize = max(12, min(25, int(remaining / 15)))
    
    result = differential_evolution(
        _negative_score_angular,
        bounds,
        maxiter=maxiter,
        tol=1e-8,
        popsize=popsize,
        seed=123,
        polish=True
    )
    
    best_params = result.x
    best_score = -result.fun
    best_poly = _angular_polygon(best_params[:20], best_params[20:40])
    
    return best_poly, best_score

def _optimize_fourier_local(initial_poly, deadline):
    """Local refinement in Fourier space using Nelder-Mead."""
    start_time = time.time()
    remaining = deadline - start_time
    if remaining < 20:
        return initial_poly, _compute_min_vertex_score(initial_poly)
    
    # Extract approximate Fourier coefficients from initial polygon
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    centroid = np.mean(initial_poly, axis=0)
    p = initial_poly - centroid
    radii = np.sqrt(np.sum(p**2, axis=1))
    mean_r = np.mean(radii)
    radii_norm = (radii - mean_r) / mean_r
    
    # Fit Fourier coefficients
    coeffs = np.zeros(8)
    for h in range(1, 5):
        coeffs[2*(h-1)] = np.mean(radii_norm * np.cos(h * angles))
        coeffs[2*(h-1)+1] = np.mean(radii_norm * np.sin(h * angles))
    
    def neg_score(coeffs):
        poly = _fourier_polygon(coeffs)
        return -_compute_min_vertex_score(poly)
    
    maxiter = max(100, int(remaining / 0.5))
    
    result = minimize(
        neg_score,
        coeffs,
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    
    best_poly = _fourier_polygon(result.x)
    best_score = -result.fun
    
    return best_poly, best_score

def _optimize_angular_local(initial_poly, deadline):
    """
    Local refinement in angular + radial space.
    Extracts initial theta and radius perturbations from the polygon.
    """
    start_time = time.time()
    remaining = deadline - start_time
    if remaining < 20:
        return initial_poly, _compute_min_vertex_score(initial_poly)
    
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    centroid = np.mean(initial_poly, axis=0)
    p = initial_poly - centroid
    
    # Extract angles and radii
    actual_angles = np.arctan2(p[:, 1], p[:, 0])
    # Unwrap angles to be continuous
    for i in range(1, n):
        while actual_angles[i] - actual_angles[i-1] > np.pi:
            actual_angles[i] -= 2 * np.pi
        while actual_angles[i] - actual_angles[i-1] < -np.pi:
            actual_angles[i] += 2 * np.pi
    
    theta_perturb = actual_angles - base_angles
    # Wrap theta_perturb to [-pi, pi]
    theta_perturb = (theta_perturb + np.pi) % (2 * np.pi) - np.pi
    
    radii = np.sqrt(np.sum(p**2, axis=1))
    mean_r = np.mean(radii)
    radius_perturb = (radii - mean_r) / mean_r
    
    initial_params = np.concatenate([theta_perturb, radius_perturb])
    
    def neg_score(params):
        poly = _angular_polygon(params[:20], params[20:40])
        return -_compute_min_vertex_score(poly)
    
    maxiter = max(200, int(remaining / 0.3))
    
    result = minimize(
        neg_score,
        initial_params,
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    
    best_poly = _angular_polygon(result.x[:20], result.x[20:40])
    best_score = -result.fun
    
    return best_poly, best_score

def _optimize_scipy_from_poly(initial_poly, deadline):
    """Refine with Nelder-Mead in Cartesian space."""
    start_time = time.time()
    remaining = deadline - start_time
    if remaining < 20:
        return initial_poly, _compute_min_vertex_score(initial_poly)
    
    best_poly = initial_poly.copy()
    best_score = _compute_min_vertex_score(best_poly)
    
    x0 = initial_poly.flatten()
    
    def neg_score(flat):
        poly = flat.reshape(20, 2)
        return -_compute_min_vertex_score(poly)
    
    maxiter = max(200, int(remaining / 0.3))
    
    result = minimize(
        neg_score,
        x0,
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    
    if time.time() < deadline:
        if -result.fun > best_score:
            best_poly = result.x.reshape(20, 2)
            best_score = -result.fun
    
    return best_poly, best_score

def _coordinate_descent(poly, deadline):
    """Fine-tune with coordinate descent."""
    start_time = time.time()
    current_poly = poly.copy()
    current_score = _compute_min_vertex_score(current_poly)
    best_poly = current_poly.copy()
    best_score = current_score
    
    step_size = 0.01
    no_improve = 0
    
    while time.time() < deadline and no_improve < 50:
        improved = False
        for i in range(20):
            if time.time() >= deadline:
                break
            old_v = current_poly[i].copy()
            for dx in [-step_size, step_size]:
                for dy in [-step_size, step_size]:
                    new_poly = current_poly.copy()
                    new_poly[i] = old_v + np.array([dx, dy])
                    score = _compute_min_vertex_score(new_poly)
                    if score > current_score + 1e-13:
                        current_poly = new_poly
                        current_score = score
                        improved = True
                        if score > best_score:
                            best_score = score
                            best_poly = current_poly.copy()
                        break
                if improved:
                    break
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            step_size *= 0.5
    
    return best_poly, best_score

def _polish_pass(poly, deadline):
    """High-precision polish using tight tolerances."""
    start_time = time.time()
    remaining = deadline - start_time
    if remaining < 10:
        return poly, _compute_min_vertex_score(poly)
    
    best_poly = poly.copy()
    best_score = _compute_min_vertex_score(best_poly)
    
    # Try small perturbations around best solution
    step = 0.001
    for _ in range(100):
        if time.time() >= deadline:
            break
        for i in range(20):
            if time.time() >= deadline:
                break
            old_v = best_poly[i].copy()
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_poly = best_poly.copy()
                    new_poly[i] = old_v + np.array([dx, dy])
                    score = _compute_min_vertex_score(new_poly)
                    if score > best_score + 1e-14:
                        best_poly = new_poly
                        best_score = score
                        break
                if best_score > _compute_min_vertex_score(poly) + 1e-14:
                    break
            if best_score > _compute_min_vertex_score(poly) + 1e-14:
                break
    
    return best_poly, best_score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin: 630 - 30 = 600
    
    # Start with regular polygon
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    # Global search in Fourier space (4 harmonics)
    if time.time() < deadline - 200:
        fe_poly, fe_score = _optimize_fourier_global(deadline)
        if fe_score > best_score:
            best_score = fe_score
            best_poly = fe_poly
    
    # Global search in angular + radial space
    if time.time() < deadline - 150:
        ag_poly, ag_score = _optimize_angular_global(deadline)
        if ag_score > best_score:
            best_score = ag_score
            best_poly = ag_poly
    
    # Local Fourier refinement
    if time.time() < deadline - 100:
        fl_poly, fl_score = _optimize_fourier_local(best_poly, deadline)
        if fl_score > best_score:
            best_score = fl_score
            best_poly = fl_poly
    
    # Local angular + radial refinement
    if time.time() < deadline - 80:
        al_poly, al_score = _optimize_angular_local(best_poly, deadline)
        if al_score > best_score:
            best_score = al_score
            best_poly = al_poly
    
    # Local refinement with scipy in Cartesian space
    if time.time() < deadline - 50:
        sc_poly, sc_score = _optimize_scipy_from_poly(best_poly, deadline)
        if sc_score > best_score:
            best_score = sc_score
            best_poly = sc_poly
    
    # Coordinate descent fine-tuning
    if time.time() < deadline - 20:
        cd_poly, cd_score = _coordinate_descent(best_poly, deadline)
        if cd_score > best_score:
            best_score = cd_score
            best_poly = cd_poly
    
    # Final polish pass
    if time.time() < deadline - 5:
        pol_poly, pol_score = _polish_pass(best_poly, deadline)
        if pol_score > best_score:
            best_score = pol_score
            best_poly = pol_poly
    
    return best_poly

# EVOLVE-BLOCK-END
