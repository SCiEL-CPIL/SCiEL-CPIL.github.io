import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_min_vertex_score(points):
    """
    Vectorized calculation of the minimum vertex score.
    points: (20, 2) array.
    Returns: float, the minimum vertex score.
    """
    n = len(points)
    
    # Normalize
    centroid = np.mean(points, axis=0)
    points_norm = points - centroid
    dists_to_centroid = np.linalg.norm(points_norm, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    points_norm = points_norm / mean_dist

    # Check strictly convex
    # 1. No repeated points
    # 2. Cyclic order and convexity
    # Calculate cross products of consecutive edges
    # Edge i: points[i] -> points[i+1]
    # Edge i+1: points[i+1] -> points[i+2]
    # Cross product z-component determines turn direction
    
    # Vectorized cross product check
    p1 = points_norm
    p2 = np.roll(points_norm, -1, axis=0)
    p3 = np.roll(points_norm, -2, axis=0)
    
    v1 = p2 - p1
    v2 = p3 - p2
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    # All cross products must have the same sign and be non-zero
    if np.any(np.abs(cross) < 1e-9):
        return 0.0
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0
        
    # Compute vertex scores
    # Distances matrix
    diffs = points_norm[:, np.newaxis, :] - points_norm[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    
    # Mask self-distances
    np.fill_diagonal(dists, np.inf)
    
    # Sort distances for each vertex
    # dists_sorted[i] contains the 19 distances from vertex i to others, sorted
    dists_sorted = np.sort(dists, axis=1)
    
    # Check d_0 < 0.05
    if np.any(dists_sorted[:, 0] < 0.05):
        return 0.0
        
    # Find k minimizing d_{k+1} - d_k
    gaps = dists_sorted[:, 1:] - dists_sorted[:, :-1]
    k_indices = np.argmin(gaps, axis=1)
    
    # D = (d_k + d_{k+1}) / 2
    # We need to gather d_k and d_{k+1} for each vertex
    row_indices = np.arange(n)
    d_k = dists_sorted[row_indices, k_indices]
    d_k1 = dists_sorted[row_indices, k_indices + 1]
    D = (d_k + d_k1) / 2.0
    
    if np.any(D < 0.1):
        return 0.0
        
    # Select 4 distances
    # idxs depend on k
    # if k == 0: [0, 1, 2, 3]
    # elif k == 17: [15, 16, 17, 18]
    # else: [k-1, k, k+1, k+2]
    
    # Initialize array for selected distances
    # Shape (20, 4)
    selected_dists = np.zeros((n, 4))
    
    mask_k0 = (k_indices == 0)
    mask_k17 = (k_indices == 17)
    mask_other = ~(mask_k0 | mask_k17)
    
    # For k=0
    if np.any(mask_k0):
        selected_dists[mask_k0, :] = dists_sorted[mask_k0, 0:4]
        
    # For k=17
    if np.any(mask_k17):
        selected_dists[mask_k17, :] = dists_sorted[mask_k17, 15:19]
        
    # For others
    if np.any(mask_other):
        k_o = k_indices[mask_other]
        # We need to gather d_{k-1}, d_k, d_{k+1}, d_{k+2}
        # Indices: k_o - 1, k_o, k_o + 1, k_o + 2
        # Create index array
        idxs = np.stack([k_o - 1, k_o, k_o + 1, k_o + 2], axis=1)
        rows = np.arange(n)[mask_other]
        selected_dists[mask_other, :] = dists_sorted[rows[:, np.newaxis], idxs]
        
    # Calculate raw score
    # ratios = max(d/D, D/d)
    # D shape: (20,) -> (20, 1)
    D_col = D[:, np.newaxis]
    ratios = np.maximum(selected_dists / D_col, D_col / selected_dists)
    raw = np.mean(ratios, axis=1)
    
    # Avoid division by zero
    raw = np.where(raw == 0, 1.0, raw)
    
    scores = np.minimum(1.0, 1.0 / raw)
    
    return np.min(scores)

def _params_to_points(params):
    """
    Convert 39 parameters to 20 points.
    params[0:19] -> angles (via softmax/cumsum)
    params[19:39] -> radii (via exp)
    """
    u = params[0:19]
    v = params[19:39]
    
    # Angles
    w = np.exp(u - np.max(u)) # For stability
    w_sum = np.sum(w)
    deltas = w / w_sum * 2 * np.pi
    angles = np.cumsum(deltas)
    # angles[0] is delta[0], so theta_0 = 0?
    # Let's define theta_0 = 0.
    # theta_i = sum(deltas[0:i])
    # My cumsum gives theta_0 = delta_0.
    # I want theta_0 = 0.
    # So I should compute cumsum of deltas[1:] for theta_1..theta_19?
    # Or just shift.
    # Let's use:
    # theta_0 = 0
    # theta_i = sum_{j=1}^i delta_j
    # So I need 19 deltas.
    # w has 19 elements.
    # deltas = w / sum(w) * 2*pi
    # theta = np.concatenate(([0], np.cumsum(deltas)))
    # This gives 20 angles.
    
    thetas = np.concatenate(([0.0], np.cumsum(deltas)))
    
    # Radii
    radii = np.exp(v)
    
    x = radii * np.cos(thetas)
    y = radii * np.sin(thetas)
    
    return np.column_stack((x, y))

def _objective(params):
    points = _params_to_points(params)
    score = _compute_min_vertex_score(points)
    if score == 0.0:
        return 100.0 # Large penalty
    return -score

def _soft_min_objective(params, beta=50.0):
    """
    Smooth approximation for local refinement.
    """
    points = _params_to_points(params)
    
    # Check convexity quickly
    n = len(points)
    centroid = np.mean(points, axis=0)
    points_norm = points - centroid
    dists_to_centroid = np.linalg.norm(points_norm, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 100.0
    points_norm = points_norm / mean_dist

    p1 = points_norm
    p2 = np.roll(points_norm, -1, axis=0)
    p3 = np.roll(points_norm, -2, axis=0)
    v1 = p2 - p1
    v2 = p3 - p2
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    if np.any(np.abs(cross) < 1e-9) or not (np.all(cross > 0) or np.all(cross < 0)):
        return 100.0

    diffs = points_norm[:, np.newaxis, :] - points_norm[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    dists_sorted = np.sort(dists, axis=1)
    
    if np.any(dists_sorted[:, 0] < 0.05):
        return 100.0
        
    gaps = dists_sorted[:, 1:] - dists_sorted[:, :-1]
    k_indices = np.argmin(gaps, axis=1)
    
    row_indices = np.arange(n)
    d_k = dists_sorted[row_indices, k_indices]
    d_k1 = dists_sorted[row_indices, k_indices + 1]
    D = (d_k + d_k1) / 2.0
    
    if np.any(D < 0.1):
        return 100.0
        
    selected_dists = np.zeros((n, 4))
    mask_k0 = (k_indices == 0)
    mask_k17 = (k_indices == 17)
    mask_other = ~(mask_k0 | mask_k17)
    
    if np.any(mask_k0):
        selected_dists[mask_k0, :] = dists_sorted[mask_k0, 0:4]
    if np.any(mask_k17):
        selected_dists[mask_k17, :] = dists_sorted[mask_k17, 15:19]
    if np.any(mask_other):
        k_o = k_indices[mask_other]
        idxs = np.stack([k_o - 1, k_o, k_o + 1, k_o + 2], axis=1)
        rows = np.arange(n)[mask_other]
        selected_dists[mask_other, :] = dists_sorted[rows[:, np.newaxis], idxs]
        
    D_col = D[:, np.newaxis]
    ratios = np.maximum(selected_dists / D_col, D_col / selected_dists)
    raw = np.mean(ratios, axis=1)
    raw = np.where(raw == 0, 1.0, raw)
    scores = np.minimum(1.0, 1.0 / raw)
    
    # Soft min
    # We want to maximize min(scores).
    # Minimize -min(scores).
    # Soft max of (-scores) is (1/beta) log sum exp(-beta scores).
    a = -beta * scores
    a_max = np.max(a)
    sum_exp = np.sum(np.exp(a - a_max))
    log_sum = a_max + np.log(sum_exp)
    obj = log_sum / beta
    return obj

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = 630.0 # Evaluator kills at 630s, keep margin
    margin = 30.0
    
    # Initial guess: Regular polygon
    # Angles: 0, 2pi/20, ..., 18*2pi/20
    # Radii: 1
    # Params: u (19) for angles, v (20) for radii
    # For regular polygon, all deltas are equal.
    # u_i = log(1) = 0.
    # v_i = log(1) = 0.
    x0 = np.zeros(39)
    
    best_params = x0
    best_score = _compute_min_vertex_score(_params_to_points(x0))
    best_points = _params_to_points(x0)
    
    # Phase 1: Global Search with Differential Evolution
    # Bounds for u: [-5, 5] (exp covers wide range)
    # Bounds for v: [-2, 2] (radii 0.1 to 7.4)
    bounds = [(-5, 5)] * 19 + [(-2, 2)] * 20
    
    if time.time() + 100 < deadline:
        try:
            res_de = differential_evolution(
                _objective,
                bounds,
                maxiter=100,
                popsize=20,
                tol=1e-6,
                update_interval=1,
                mutation=(0.5, 1.5),
                recombination=0.9,
                seed=42
            )
            if res_de.fun < -best_score:
                best_score = -res_de.fun
                best_params = res_de.x
                best_points = _params_to_points(best_params)
        except Exception as e:
            pass

    # Phase 2: Local Refinement with L-BFGS-B using soft-min
    # Try a few restarts/perturbations
    rng = np.random.default_rng(42)
    
    restarts = [best_params]
    for i in range(5):
        if time.time() > deadline - 50:
            break
        # Perturb best solution
        x_pert = best_params + rng.normal(0, 0.1, size=39)
        restarts.append(x_pert)
        
    current_beta = 10.0
    for x_init in restarts:
        if time.time() > deadline - 20:
            break
        try:
            # Annealing beta
            for beta in [5.0, 10.0, 20.0, 50.0, 100.0]:
                if time.time() > deadline - 10:
                    break
                res_lm = minimize(
                    _soft_min_objective,
                    x_init,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                x_init = res_lm.x
                # Evaluate exact score
                score_exact = _compute_min_vertex_score(_params_to_points(x_init))
                if score_exact > best_score:
                    best_score = score_exact
                    best_params = x_init
                    best_points = _params_to_points(x_init)
        except Exception as e:
            pass

    # Phase 3: Final Polish with Nelder-Mead on exact objective
    if time.time() + 10 < deadline:
        try:
            res_nm = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            score_nm = _compute_min_vertex_score(_params_to_points(res_nm.x))
            if score_nm > best_score:
                best_score = score_nm
                best_params = res_nm.x
                best_points = _params_to_points(best_params)
        except Exception as e:
            pass

    return best_points
