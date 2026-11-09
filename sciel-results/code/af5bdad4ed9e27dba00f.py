import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def params_to_weights(params):
    """
    Convert 19 parameters to 20 positive weights.
    The 20th weight is fixed to 1.0 to remove scale redundancy.
    """
    w_full = np.append(params**2 + 0.01, 1.0)
    w_full = w_full / np.sum(w_full)
    return w_full * 2 * np.pi

def get_points(params):
    """
    params: 19 values. Returns (20, 2) array of points on unit circle.
    """
    w = params_to_weights(params)
    angles = np.cumsum(w)
    angles = angles % (2 * np.pi)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def fourier_params_to_weights(fourier_params):
    """
    Convert Fourier coefficients (a1, b1, a2, b2, a3, b3) to 20 positive weights.
    This restricts the angular deviations to low-frequency modes for smoothness.
    """
    n = 20
    # Base angle for regular polygon
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Fourier series: delta_theta = sum (a_k * cos(k*theta) + b_k * sin(k*theta))
    # Here theta = 2*pi*j/n
    a1, b1, a2, b2, a3, b3 = fourier_params
    
    deviations = np.zeros(n)
    for j in range(n):
        theta = 2 * np.pi * j / n
        deviations[j] = (a1 * np.cos(theta) + b1 * np.sin(theta) +
                         a2 * np.cos(2*theta) + b2 * np.sin(2*theta) +
                         a3 * np.cos(3*theta) + b3 * np.sin(3*theta))
    
    # Ensure positive weights by using exp of deviations scaled
    # We want angles = base_angles + deviations, but we need to ensure they are in [0, 2*pi)
    # Actually, let's just compute the angles directly and then derive weights from them
    angles = base_angles + deviations
    # Sort angles to ensure cyclic order (though they should be sorted already if deviations are small)
    # To be safe, let's just use the angles as is, but we need to ensure they are increasing
    # If deviations are small, the order is preserved.
    
    # Compute weights from angles: w_j = angle_{j+1} - angle_j (with wrap-around)
    angles = np.sort(angles)  # Ensure sorted
    # Add wrap-around
    angles_full = np.append(angles, angles[0] + 2 * np.pi)
    weights = np.diff(angles_full)
    
    # Ensure weights are positive (they should be if deviations are small)
    weights = np.maximum(weights, 1e-6)
    weights = weights / np.sum(weights) * 2 * np.pi
    
    return weights

def vertex_scores_all(points):
    """
    Compute vertex scores for all 20 vertices. Returns array of 20 scores.
    """
    n = 20
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    scores = np.zeros(n)
    for i in range(n):
        d = dists[i, :].copy()
        d[i] = np.inf
        sorted_d = np.sort(d)
        if sorted_d[0] < 0.05:
            scores[i] = 0.0
            continue
        gaps = sorted_d[1:] - sorted_d[:-1]
        k = np.argmin(gaps)
        D = (sorted_d[k] + sorted_d[k+1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
        if k == 0:
            d_sel = sorted_d[0:4]
        elif k == 17:
            d_sel = sorted_d[15:19]
        else:
            d_sel = sorted_d[k-1:k+3]
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
    return scores

def soft_min(scores, beta=30.0):
    """
    Smooth minimum using log-sum-exp.
    """
    scores = np.asarray(scores, dtype=float)
    m = np.min(scores)
    return m - np.log(np.sum(np.exp(-beta * (scores - m)))) / beta

def min_vertex_score(params):
    points = get_points(params)
    scores = vertex_scores_all(points)
    return -soft_min(scores, beta=30.0)

def exact_min_vertex_score(params):
    points = get_points(params)
    scores = vertex_scores_all(points)
    return -np.min(scores)

def min_vertex_score_fourier(fourier_params):
    w = fourier_params_to_weights(fourier_params)
    angles = np.cumsum(w)
    angles = angles % (2 * np.pi)
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.column_stack((x, y))
    scores = vertex_scores_all(points)
    return -soft_min(scores, beta=30.0)

def exact_min_vertex_score_fourier(fourier_params):
    w = fourier_params_to_weights(fourier_params)
    angles = np.cumsum(w)
    angles = angles % (2 * np.pi)
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.column_stack((x, y))
    scores = vertex_scores_all(points)
    return -np.min(scores)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590
    
    # Initial guess: regular polygon -> all 19 free params equal
    x0 = np.ones(19)
    
    best_params = x0.copy()
    best_score = -exact_min_vertex_score(x0)
    best_fourier_params = None
    best_fourier_score = -np.inf
    
    # Phase 0: Fourier-based search (6D optimization)
    # This is a lower-dimensional, smoother search space
    if time.time() < deadline - 300:
        try:
            # Fourier parameters: a1, b1, a2, b2, a3, b3
            # Small deviations from regular polygon
            bounds_fourier = [(-0.1, 0.1) for _ in range(6)]
            
            # Differential Evolution in Fourier space
            res_de_f = differential_evolution(
                min_vertex_score_fourier,
                bounds_fourier,
                maxiter=300,
                popsize=20,
                tol=1e-8,
                update_interval=1,
                seed=42
            )
            s_f = -exact_min_vertex_score_fourier(res_de_f.x)
            if s_f > best_fourier_score:
                best_fourier_score = s_f
                best_fourier_params = res_de_f.x.copy()
            
            # Local refinement in Fourier space
            rng_f = np.random.default_rng(42)
            restarts_f = [best_fourier_params.copy()]
            for i in range(10):
                x_rand_f = rng_f.uniform(-0.1, 0.1, size=6)
                restarts_f.append(x_rand_f)
                x_pert_f = best_fourier_params + rng_f.normal(0, 0.02, size=6)
                x_pert_f = np.clip(x_pert_f, -0.1, 0.1)
                restarts_f.append(x_pert_f)
            
            for x_init_f in restarts_f:
                if time.time() > deadline - 200:
                    break
                try:
                    res_lm_f = minimize(
                        min_vertex_score_fourier,
                        x_init_f,
                        method='L-BFGS-B',
                        bounds=bounds_fourier,
                        options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    s_f = -exact_min_vertex_score_fourier(res_lm_f.x)
                    if s_f > best_fourier_score:
                        best_fourier_score = s_f
                        best_fourier_params = res_lm_f.x.copy()
                except Exception:
                    pass
        except Exception:
            pass
    
    # Phase 1: Differential Evolution with soft-min objective (19D)
    if time.time() < deadline - 200:
        try:
            bounds = [(0.1, 10.0) for _ in range(19)]
            res_de = differential_evolution(
                min_vertex_score,
                bounds,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                update_interval=1,
                seed=42
            )
            s = -exact_min_vertex_score(res_de.x)
            if s > best_score:
                best_score = s
                best_params = res_de.x.copy()
        except Exception:
            pass

    # Phase 2: Local Refinement with L-BFGS-B on soft-min
    rng = np.random.default_rng(42)
    
    restarts = [best_params.copy()]
    for i in range(20):
        if time.time() > deadline - 50:
            break
        x_rand = rng.uniform(0.5, 2.0, size=19)
        restarts.append(x_rand)
        x_pert = best_params + rng.normal(0, 0.1, size=19)
        x_pert = np.clip(x_pert, 0.1, 10.0)
        restarts.append(x_pert)
        
    for x_init in restarts:
        if time.time() > deadline - 20:
            break
        try:
            res_lm = minimize(
                min_vertex_score,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            s = -exact_min_vertex_score(res_lm.x)
            if s > best_score:
                best_score = s
                best_params = res_lm.x.copy()
        except Exception:
            pass

    # Phase 3: Multiple Nelder-Mead restarts on exact objective
    if time.time() < deadline - 10:
        nm_starts = [best_params.copy()]
        for i in range(10):
            if time.time() > deadline - 10:
                break
            x_pert = best_params + rng.normal(0, 0.05, size=19)
            x_pert = np.clip(x_pert, 0.1, 10.0)
            nm_starts.append(x_pert)
        
        for x_init in nm_starts:
            if time.time() > deadline - 5:
                break
            try:
                res_nm = minimize(
                    exact_min_vertex_score,
                    x_init,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                s = -exact_min_vertex_score(res_nm.x)
                if s > best_score:
                    best_score = s
                    best_params = res_nm.x.copy()
            except Exception:
                pass
    
    # Choose the best result between 19D and 6D Fourier search
    if best_fourier_params is not None and best_fourier_score > best_score:
        w = fourier_params_to_weights(best_fourier_params)
        angles = np.cumsum(w)
        angles = angles % (2 * np.pi)
        x = np.cos(angles)
        y = np.sin(angles)
        points = np.column_stack((x, y))
    else:
        points = get_points(best_params)
    
    # Ensure validity: check that the polygon is strictly convex
    # Check for repeated points
    for i in range(20):
        for j in range(i+1, 20):
            if np.linalg.norm(points[i] - points[j]) < 1e-6:
                # Fallback to regular polygon
                angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
                points = np.column_stack((np.cos(angles), np.sin(angles)))
                break
    
    # Normalize: subtract centroid, divide by mean distance
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist > 0.01:
        points = points / mean_dist
    
    return points
