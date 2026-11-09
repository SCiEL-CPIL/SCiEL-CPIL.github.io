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

def _regular_polygon():
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.stack([x, y], axis=1)

def _fourier_polygon(fourier_coeffs):
    """
    Reconstruct polygon from Fourier coefficients.
    Supports harmonics 1, 2, and 3.
    coeffs: [a1, b1, a2, b2, a3, b3]
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = np.ones(n)
    if len(fourier_coeffs) >= 2:
        radii += fourier_coeffs[0] * np.cos(base_angles) + fourier_coeffs[1] * np.sin(base_angles)
    if len(fourier_coeffs) >= 4:
        radii += fourier_coeffs[2] * np.cos(2 * base_angles) + fourier_coeffs[3] * np.sin(2 * base_angles)
    if len(fourier_coeffs) >= 6:
        radii += fourier_coeffs[4] * np.cos(3 * base_angles) + fourier_coeffs[5] * np.sin(3 * base_angles)
    
    x = radii * np.cos(base_angles)
    y = radii * np.sin(base_angles)
    return np.stack([x, y], axis=1)

def _negative_score_fourier(fourier_coeffs):
    poly = _fourier_polygon(fourier_coeffs)
    score = _compute_min_vertex_score(poly)
    return -score

def _optimize_fourier_global():
    """Use differential evolution to find good Fourier coefficients."""
    start_time = time.time()
    deadline = start_time + 150.0
    
    # Bounds for a1, b1, a2, b2, a3, b3
    bounds = [(-0.3, 0.3), (-0.3, 0.3), (-0.2, 0.2), (-0.2, 0.2), (-0.1, 0.1), (-0.1, 0.1)]
    
    try:
        result = differential_evolution(
            _negative_score_fourier,
            bounds,
            maxiter=80,
            tol=1e-8,
            popsize=20,
            seed=42,
            polish=True
        )
        
        if time.time() > deadline:
            return _regular_polygon(), 0.0
        
        best_fourier = result.x
        best_score = -result.fun
        best_poly = _fourier_polygon(best_fourier)
        
        return best_poly, best_score
    except Exception:
        return _regular_polygon(), 0.0

def _optimize_fourier_local(initial_coeffs, max_time=60.0):
    """Refine Fourier coefficients using Nelder-Mead."""
    start_time = time.time()
    
    def neg_score(coeffs):
        return _negative_score_fourier(coeffs)
    
    try:
        result = minimize(
            neg_score,
            initial_coeffs,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
        )
        
        if time.time() > start_time + max_time:
            return initial_coeffs, -_negative_score_fourier(initial_coeffs)
            
        best_coeffs = result.x
        best_score = -result.fun
        return best_coeffs, best_score
    except Exception:
        return initial_coeffs, -_negative_score_fourier(initial_coeffs)

def _optimize_scipy_from_poly(initial_poly, max_time=100.0):
    """Refine with Nelder-Mead in Cartesian space."""
    start_time = time.time()
    best_poly = initial_poly.copy()
    best_score = _compute_min_vertex_score(best_poly)
    
    x0 = initial_poly.flatten()
    
    def neg_score(flat):
        poly = flat.reshape(20, 2)
        return -_compute_min_vertex_score(poly)
    
    try:
        result = minimize(
            neg_score,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
        )
        
        if time.time() < start_time + max_time:
            if result.fun < -best_score:
                best_poly = result.x.reshape(20, 2)
                best_score = -result.fun
    except Exception:
        pass

    return best_poly, best_score

def _coordinate_descent(poly, max_time=40.0):
    """Fine-tune with coordinate descent."""
    start_time = time.time()
    current_poly = poly.copy()
    current_score = _compute_min_vertex_score(current_poly)
    best_poly = current_poly.copy()
    best_score = current_score
    
    step_size = 0.01
    no_improve = 0
    
    while time.time() - start_time < max_time and no_improve < 30:
        improved = False
        for i in range(20):
            if time.time() - start_time > max_time:
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

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0
    
    # Start with regular polygon
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    # 1. Global search in Fourier space (6D)
    if time.time() < deadline - 250:
        fe_poly, fe_score = _optimize_fourier_global()
        if fe_score > best_score:
            best_score = fe_score
            best_poly = fe_poly
            # Recover coefficients for local Fourier refinement
            # We need to reconstruct coefficients from the polygon or keep them from DE
            # DE returns the coefficients directly, so we should modify _optimize_fourier_global to return coeffs too
            # For now, let's just use the polygon. But local Fourier refinement is better.
            # Let's re-run a quick local Fourier refinement on the best found coeffs.
            # Actually, let's change _optimize_fourier_global to return coeffs.
            
    # Let's refine the strategy:
    # 1. DE finds best_coeffs and best_poly.
    # 2. Local Nelder-Mead on best_coeffs.
    # 3. Local Nelder-Mead on best_poly (Cartesian).
    # 4. Coordinate Descent on best_poly.
    
    # Re-implementing flow to be safer:
    
    # Phase 1: Global Fourier
    start_p1 = time.time()
    bounds = [(-0.3, 0.3), (-0.3, 0.3), (-0.2, 0.2), (-0.2, 0.2), (-0.1, 0.1), (-0.1, 0.1)]
    best_coeffs = np.zeros(6)
    try:
        res_de = differential_evolution(
            _negative_score_fourier,
            bounds,
            maxiter=80,
            tol=1e-8,
            popsize=20,
            seed=42,
            polish=True
        )
        if time.time() - start_p1 < 150:
            best_coeffs = res_de.x
            best_score_fe = -res_de.fun
            best_poly_fe = _fourier_polygon(best_coeffs)
            if best_score_fe > best_score:
                best_score = best_score_fe
                best_poly = best_poly_fe
    except Exception:
        pass

    # Phase 2: Local Fourier Refinement
    if time.time() < deadline - 150:
        start_p2 = time.time()
        coeffs_refined, score_refined = _optimize_fourier_local(best_coeffs, max_time=60.0)
        poly_refined = _fourier_polygon(coeffs_refined)
        if score_refined > best_score:
            best_score = score_refined
            best_poly = poly_refined

    # Phase 3: Local Cartesian Refinement
    if time.time() < deadline - 80:
        start_p3 = time.time()
        sc_poly, sc_score = _optimize_scipy_from_poly(best_poly, max_time=80.0)
        if sc_score > best_score:
            best_score = sc_score
            best_poly = sc_poly

    # Phase 4: Coordinate Descent
    if time.time() < deadline - 10:
        cd_poly, cd_score = _coordinate_descent(best_poly, max_time=40.0)
        if cd_score > best_score:
            best_score = cd_score
            best_poly = cd_poly

    return best_poly

# EVOLVE-BLOCK-END
