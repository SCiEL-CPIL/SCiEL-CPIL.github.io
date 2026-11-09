import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_all_scores(polygon):
    """Compute per-vertex scores for all vertices of polygon."""
    n = len(polygon)
    scores = np.zeros(n)
    # Vectorized pairwise distances
    diff = polygon[:, np.newaxis, :] - polygon[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    for i in range(n):
        d = np.sort(dists[i])
        if d[0] < 0.05:
            scores[i] = 0.0
            continue
        gaps = np.diff(d)
        k = int(np.argmin(gaps))
        D = (d[k] + d[k+1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
        if k == 0:
            selected = d[0:4]
        elif k == 17:
            selected = d[15:19]
        else:
            selected = d[k-1:k+3]
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
    return scores

def _polygon_min_score(polygon):
    """Compute the minimum vertex score over all vertices."""
    return float(np.min(_compute_all_scores(polygon)))

def _normalize(polygon):
    """Normalize polygon: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(polygon, axis=0)
    poly = polygon - centroid
    dists = np.sqrt(np.sum(poly**2, axis=1))
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return poly
    return poly / mean_dist

def _is_strictly_convex(polygon):
    """Check if polygon is strictly convex with correct total turning."""
    n = len(polygon)
    if n < 3:
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-10:
                return False
    # Compute cross products of consecutive edges
    cross_prods = []
    for i in range(n):
        v1 = polygon[(i+1) % n] - polygon[i]
        v2 = polygon[(i+2) % n] - polygon[(i+1) % n]
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        cross_prods.append(cross)
    cross_prods = np.array(cross_prods)
    if np.any(np.abs(cross_prods) < 1e-12):
        return False
    if np.any(cross_prods > 0) and np.any(cross_prods < 0):
        return False
    # Total turning check
    edge_angles = []
    for i in range(n):
        v = polygon[(i+1) % n] - polygon[i]
        edge_angles.append(np.arctan2(v[1], v[0]))
    turns = []
    for i in range(n):
        a1 = edge_angles[i]
        a2 = edge_angles[(i+1) % n]
        turn = a2 - a1
        while turn > np.pi:
            turn -= 2*np.pi
        while turn <= -np.pi:
            turn += 2*np.pi
        turns.append(turn)
    total_turn = sum(turns)
    if abs(abs(total_turn) - 2*np.pi) > 1e-6:
        return False
    return True

def _build_polygon_from_radii(radii, angles):
    """Build polygon from radii and fixed angles."""
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _build_polygon_fourier(params, n=20):
    """Build polygon from radial Fourier modulation parameters.
    params: [a0, a1c, a1s, a2c, a2s, ..., a5c, a5s]
    r(theta) = a0 + sum_{m=1}^{5} (a_m*cos(m*theta) + b_m*sin(m*theta))
    """
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    r = np.full(n, params[0])
    idx = 1
    for m in range(1, 6):
        a_m = params[idx]
        b_m = params[idx+1]
        r += a_m * np.cos(m * theta) + b_m * np.sin(m * theta)
        idx += 2
    r = np.maximum(r, 0.05)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)

def _objective_fourier(params, n=20):
    """Negative min vertex score for minimization using Fourier representation."""
    polygon = _build_polygon_fourier(params, n)
    polygon = _normalize(polygon)
    if not _is_strictly_convex(polygon):
        return 10.0
    return -_polygon_min_score(polygon)

def _objective_radii(radii, angles, n=20):
    """Negative min vertex score for minimization using direct radii."""
    polygon = _build_polygon_from_radii(radii, angles)
    polygon = _normalize(polygon)
    if not _is_strictly_convex(polygon):
        return 10.0
    return -_polygon_min_score(polygon)

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon maximizing the minimum per-vertex equidistance score.
    Uses differential evolution in Fourier space for global search,
    then Nelder-Mead for local refinement.
    """
    start_time = time.time()
    deadline = start_time + 570.0  # safety margin before 630s limit
    n = 20
    
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Regular polygon baseline
    best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    best_polygon = _normalize(best_polygon)
    best_score = _polygon_min_score(best_polygon)
    
    np.random.seed(12345)
    
    # Strategy 1: Differential Evolution in Fourier parameter space (global search)
    # Use 4 Fourier harmonics for a balance of expressiveness and searchability
    n_harmonics = 4
    n_params = 1 + 2 * n_harmonics  # a0 + n_harmonics*(a_m, b_m)
    
    # Define bounds for Fourier parameters
    bounds = [(0.5, 1.5)]  # a0
    for _ in range(n_harmonics):
        bounds.append((-0.3, 0.3))  # a_m
        bounds.append((-0.3, 0.3))  # b_m
    
    def _de_objective(params):
        if time.time() > deadline - 30:
            return 10.0
        polygon = _build_polygon_fourier(params, n)
        polygon = _normalize(polygon)
        if not _is_strictly_convex(polygon):
            return 10.0
        return -_polygon_min_score(polygon)
    
    try:
        de_result = differential_evolution(
            _de_objective, bounds,
            seed=42, maxiter=200, popsize=15,
            tol=1e-10, polish=True,
            init='sobol'
        )
        if de_result.fun < -best_score:
            best_score = -de_result.fun
            best_polygon = _normalize(_build_polygon_fourier(de_result.x, n))
    except Exception:
        pass
    
    # Strategy 2: Nelder-Mead refinement in Fourier space from DE result
    if time.time() < deadline - 120:
        try:
            start_params = de_result.x.copy() if hasattr(de_result, 'x') else np.zeros(n_params)
            if not hasattr(de_result, 'x') or de_result.fun >= -best_score:
                start_params = np.zeros(n_params)
                start_params[0] = 1.0
            
            result = minimize(
                _objective_fourier, start_params.copy(), method='Nelder-Mead',
                args=(n,), options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < -best_score:
                best_score = -result.fun
                best_polygon = _normalize(_build_polygon_fourier(result.x, n))
        except Exception:
            pass
    
    # Strategy 3: Multiple random restarts with Fourier Nelder-Mead
    for restart in range(8):
        if time.time() > deadline - 90:
            break
        params = np.zeros(n_params)
        params[0] = 1.0
        scale = 0.2 / (1 + restart * 0.4)
        for i in range(1, n_params):
            params[i] += np.random.uniform(-scale, scale)
        
        try:
            result = minimize(
                _objective_fourier, params, method='Nelder-Mead',
                args=(n,), options={'maxiter': 4000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < -best_score:
                best_score = -result.fun
                best_polygon = _normalize(_build_polygon_fourier(result.x, n))
        except Exception:
            pass
    
    # Strategy 4: Direct radius optimization with Nelder-Mead for local refinement
    if time.time() < deadline - 60:
        bp = best_polygon
        bp_radii = np.sqrt(np.sum(bp**2, axis=1))
        current_radii = bp_radii.copy()
        
        try:
            result = minimize(
                _objective_radii, current_radii.copy(), method='Nelder-Mead',
                args=(angles, n), options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < -best_score:
                best_score = -result.fun
                best_polygon = _normalize(_build_polygon_from_radii(result.x, angles))
        except Exception:
            pass
    
    # Strategy 5: Coordinate descent on radii for fine-tuning
    if time.time() < deadline - 45:
        bp = best_polygon
        current_radii = np.sqrt(np.sum(bp**2, axis=1)).copy()
        
        for outer_iter in range(40):
            if time.time() > deadline - 30:
                break
            improved = False
            base_step = 0.015 * (0.93 ** outer_iter)
            
            for i in range(n):
                if time.time() > deadline - 30:
                    break
                best_r = current_radii[i]
                best_s = _objective_radii(current_radii, angles, n)
                for delta in [base_step, -base_step, base_step*0.5, -base_step*0.5,
                              base_step*2.0, -base_step*2.0]:
                    trial = current_radii.copy()
                    trial[i] = max(0.05, current_radii[i] + delta)
                    trial_s = _objective_radii(trial, angles, n)
                    if trial_s < best_s - 1e-14:
                        best_s = trial_s
                        best_r = trial[i]
                if abs(best_r - current_radii[i]) > 1e-12:
                    current_radii[i] = best_r
                    improved = True
            
            if not improved:
                break
            
            s = _objective_radii(current_radii, angles, n)
            if s < -best_score:
                best_score = -s
                best_polygon = _normalize(_build_polygon_from_radii(current_radii, angles))
    
    # Final verification
    final_polygon = _normalize(best_polygon)
    if not _is_strictly_convex(final_polygon):
        # Fallback to regular polygon
        final_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        final_polygon = _normalize(final_polygon)
    
    return final_polygon
