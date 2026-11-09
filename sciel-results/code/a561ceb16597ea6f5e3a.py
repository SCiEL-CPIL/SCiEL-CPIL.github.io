import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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
    
    # Compute angles directly
    angles = base_angles + deviations
    
    # Ensure sorted order (should be preserved for small deviations)
    angles = np.sort(angles)
    
    # Compute weights from angles: w_j = angle_{j+1} - angle_j (with wrap-around)
    angles_full = np.append(angles, angles[0] + 2 * np.pi)
    weights = np.diff(angles_full)
    
    # Ensure weights are positive
    weights = np.maximum(weights, 1e-6)
    weights = weights / np.sum(weights) * 2 * np.pi
    
    return weights

def get_points_fourier(fourier_params):
    """
    Returns (20, 2) array of points on unit circle from Fourier parameters.
    """
    w = fourier_params_to_weights(fourier_params)
    angles = np.cumsum(w)
    angles = angles % (2 * np.pi)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

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
        
        # Check validity: no repeated points (distance > 0.05)
        if sorted_d[0] < 0.05:
            scores[i] = 0.0
            continue
            
        gaps = sorted_d[1:] - sorted_d[:-1]
        k = np.argmin(gaps)
        D = (sorted_d[k] + sorted_d[k+1]) / 2.0
        
        # Check validity: D must be > 0.1
        if D < 0.1:
            scores[i] = 0.0
            continue
            
        # Select the 4 distances around the gap
        if k == 0:
            d_sel = sorted_d[0:4]
        elif k == 17:
            d_sel = sorted_d[15:19]
        else:
            d_sel = sorted_d[k-1:k+3]
            
        # Calculate raw score
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
    # Numerical stability
    if m < -1e10:
        return m
    exp_terms = np.exp(-beta * (scores - m))
    # Clip to avoid overflow in sum
    exp_terms = np.clip(exp_terms, 0, 1e10)
    return m - np.log(np.sum(exp_terms)) / beta

def exact_min_vertex_score_fourier(fourier_params):
    """
    Objective function: negative of the minimum vertex score.
    """
    points = get_points_fourier(fourier_params)
    
    # Quick convexity check: ensure no repeated points and consistent turning
    # If points are very close, scores will be 0, but let's ensure we don't crash
    # The vertex_scores_all function handles the 0.05 check which implies validity for scoring purposes
    # However, the task requires strict convexity. 
    # If the polygon is not convex, the "cyclic order" might be broken or self-intersecting.
    # For small deviations from regular polygon, it remains convex.
    # We rely on the optimization to stay in the convex region.
    
    scores = vertex_scores_all(points)
    return -np.min(scores)

def min_vertex_score_fourier(fourier_params, beta=30.0):
    """
    Smooth objective function for optimization.
    """
    points = get_points_fourier(fourier_params)
    scores = vertex_scores_all(points)
    return -soft_min(scores, beta=beta)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit
    
    # Parameters: [a1, b1, a2, b2, a3, b3]
    # Bounds: Small deviations to maintain convexity and equidistance properties
    bounds = [(-0.3, 0.3) for _ in range(6)]
    
    best_coeffs = np.zeros(6)
    best_score = -exact_min_vertex_score_fourier(best_coeffs)
    
    # Initial regular polygon score
    # For regular polygon, all distances are symmetric.
    # d_0 = 2*sin(pi/20), etc.
    # Score should be close to 1.
    
    # Phase 1: Differential Evolution in Fourier space
    # Global search to find good basins
    if time.time() < deadline - 200:
        try:
            res_de = differential_evolution(
                min_vertex_score_fourier,
                bounds,
                args=(10.0,),  # Start with lower beta for smoother landscape
                maxiter=100,
                popsize=20,
                tol=1e-8,
                update_interval=1,
                seed=42
            )
            s = -exact_min_vertex_score_fourier(res_de.x)
            if s > best_score:
                best_score = s
                best_coeffs = res_de.x.copy()
        except Exception:
            pass

    # Phase 2: Multi-start Local Refinement with increasing Beta (Annealing)
    # This is the key improvement: start smooth, end sharp.
    rng = np.random.default_rng(42)
    
    # Generate a pool of starting points
    starts = [best_coeffs.copy()]
    
    # Random starts
    for i in range(30):
        if time.time() > deadline - 100:
            break
        x_rand = rng.uniform(-0.2, 0.2, size=6)
        starts.append(x_rand)
        
    # Perturbed best
    for i in range(10):
        if time.time() > deadline - 100:
            break
        x_pert = best_coeffs + rng.normal(0, 0.05, size=6)
        x_pert = np.clip(x_pert, -0.3, 0.3)
        starts.append(x_pert)
        
    # Beta progression: 10 -> 20 -> 50 -> 100
    betas = [10.0, 20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline - 50:
            break
            
        # For each start, try to optimize with current beta
        for x_init in starts:
            if time.time() > deadline - 50:
                break
            try:
                res_lm = minimize(
                    min_vertex_score_fourier,
                    x_init,
                    method='L-BFGS-B',
                    bounds=bounds,
                    args=(beta,),
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                s = -exact_min_vertex_score_fourier(res_lm.x)
                if s > best_score:
                    best_score = s
                    best_coeffs = res_lm.x.copy()
                    # Add the improved solution to starts for the next beta
                    starts.append(res_lm.x.copy())
            except Exception:
                pass
        
        # After each beta stage, we might want to re-seed starts with the current best
        # to ensure we don't get stuck in a local min that is good for low beta but bad for high beta
        # But adding the best found so far is crucial.

    # Phase 3: High-precision polish on the exact objective
    # Use Nelder-Mead or L-BFGS-B with very tight tolerances on the exact (non-smooth) objective
    # Since the exact objective is piecewise smooth, L-BFGS-B might struggle with gradients at kinks.
    # Nelder-Mead is derivative-free and robust.
    
    if time.time() < deadline - 10:
        # Start from the best found
        x_init = best_coeffs.copy()
        try:
            res_nm = minimize(
                exact_min_vertex_score_fourier,
                x_init,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            s = -exact_min_vertex_score_fourier(res_nm.x)
            if s > best_score:
                best_score = s
                best_coeffs = res_nm.x.copy()
        except Exception:
            pass
        
        # Try a few more Nelder-Mead restarts around the best
        for i in range(5):
            if time.time() > deadline - 5:
                break
            x_pert = best_coeffs + rng.normal(0, 0.01, size=6)
            x_pert = np.clip(x_pert, -0.3, 0.3)
            try:
                res_nm = minimize(
                    exact_min_vertex_score_fourier,
                    x_pert,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                s = -exact_min_vertex_score_fourier(res_nm.x)
                if s > best_score:
                    best_score = s
                    best_coeffs = res_nm.x.copy()
            except Exception:
                pass

    # Generate final points
    points = get_points_fourier(best_coeffs)
    
    # Normalize: subtract centroid, divide by mean distance
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists)
    
    if mean_dist <= 0.01:
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
    else:
        points = points / mean_dist
        
    return points
