# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def _compute_vertex_score(vertices):
    """
    Compute the per-vertex score for a single vertex of a polygon.
    vertices: (n, 2) array of polygon vertices in cyclic order.
    Returns the score for vertex 0 (we rotate so the target vertex is index 0).
    """
    n = len(vertices)
    # Distances from vertex 0 to all others
    diffs = vertices - vertices[0]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    # Exclude self
    other_dists = dists[1:]
    # Sort
    d = np.sort(other_dists)
    # Check min distance
    if d[0] < 0.05:
        return 0.0
    # Compute gaps
    gaps = d[1:] - d[:-1]
    # Find first index minimizing the gap d_{k+1} - d_k
    # k is index in d, so we look at gaps[k] = d[k+1] - d[k]
    k = np.argmin(gaps)
    D = (d[k] + d[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    # Select four distances
    if k == 0:
        sel = d[0:4]
    elif k == 17:
        sel = d[15:19]
    else:
        sel = d[k-1:k+3]
    raw = np.mean(np.maximum(sel / D, D / sel))
    score = min(1.0, 1.0 / raw)
    return score

def _polygon_min_score(polygon):
    """
    Compute the minimum vertex score over all vertices of a polygon.
    polygon: (n, 2) array.
    """
    n = len(polygon)
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    # Check convexity
    # No repeated points
    if np.any(np.linalg.norm(np.diff(np.vstack([normalized, normalized[:1]]), axis=0), axis=1) < 1e-10):
        return 0.0
    # Compute cross products for turns
    edges = np.diff(np.vstack([normalized, normalized[:1]]), axis=0)
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    # Check all turns are same sign and nonzero
    if np.any(np.abs(cross) < 1e-10):
        return 0.0
    # Check all same sign
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return 0.0
    # Total turning should be +-2*pi
    total_turn = np.sum(np.arctan2(cross, edges[:, 0] * edges[:, 0] + edges[:, 1] * edges[:, 1]))
    if abs(abs(total_turn) - 2 * np.pi) > 1e-6:
        return 0.0
    
    # Compute min vertex score
    min_score = 1.0
    for i in range(n):
        # Rotate so vertex i is at index 0
        rotated = np.roll(normalized, -i, axis=0)
        v_score = _compute_vertex_score(rotated)
        if v_score < min_score:
            min_score = v_score
    return min_score

def _params_to_polygon(params):
    """
    Convert parameter vector to polygon vertices.
    params: [angle_amplitudes (10), radius_amplitudes (10)]
    We use a low-frequency Fourier series to ensure smoothness and convexity.
    n = 20.
    """
    n = 20
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Extract parameters
    angle_params = params[:10]
    radius_params = params[10:20]
    
    # Reconstruct angle perturbations
    # We use cos and sin terms for k=1..5 (since n=20, we can go higher, but low freq is safer for convexity)
    # Actually, let's use up to k=5 for both angle and radius to allow some flexibility but maintain smoothness.
    # Angle: theta_i = base_angles_i + sum_{k=1}^{5} (a_k cos(k * base_angles_i) + b_k sin(k * base_angles_i))
    # Radius: r_i = 1.0 + sum_{k=1}^{5} (c_k cos(k * base_angles_i) + d_k sin(k * base_angles_i))
    
    theta_pert = np.zeros(n)
    r_pert = np.zeros(n)
    
    for k in range(1, 6):
        a_k = angle_params[2*(k-1)]
        b_k = angle_params[2*(k-1)+1]
        c_k = radius_params[2*(k-1)]
        d_k = radius_params[2*(k-1)+1]
        
        theta_pert += a_k * np.cos(k * base_angles) + b_k * np.sin(k * base_angles)
        r_pert += c_k * np.cos(k * base_angles) + d_k * np.sin(k * base_angles)
    
    # Ensure radius stays positive (convexity requirement)
    # Base radius 1.0, perturbation should be small. 
    # If we want to be safe, we can clip r to [0.5, 1.5] or similar.
    # However, clipping introduces non-differentiability. 
    # Instead, we can use a softplus or just assume small params in the optimization bounds.
    # For Nelder-Mead, bounds are not directly enforced, so we must rely on the objective penalty.
    # Let's add a small constant to ensure r > 0.
    radii = 1.0 + r_pert
    
    # If any radius is too small, the polygon might not be strictly convex or might have close points.
    # We will penalize this in the objective.
    
    x = radii * np.cos(base_angles + theta_pert)
    y = radii * np.sin(base_angles + theta_pert)
    
    return np.stack([x, y], axis=1)

def _objective(params):
    """
    Objective function for optimization.
    We want to maximize the minimum vertex score.
    We use a soft-min (log-sum-exp) of the negative scores to make the objective smooth.
    score_i = _compute_vertex_score for vertex i.
    We want max(min(score_i)).
    Let f_i = -log(score_i) if score_i > 0, else large penalty.
    Then min(score_i) = exp(-max(f_i)).
    Maximizing min(score_i) is equivalent to minimizing max(f_i).
    A smooth approximation to max(f_i) is (1/beta) * log(sum(exp(beta * f_i))).
    """
    polygon = _params_to_polygon(params)
    
    # Check convexity and validity first
    # We can do a quick check here to avoid expensive score computation if invalid
    n = 20
    edges = np.diff(np.vstack([polygon, polygon[:1]]), axis=0)
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    
    if np.any(np.abs(cross) < 1e-6):
        return 100.0
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return 100.0
    
    # Check no repeated points
    if np.any(np.linalg.norm(edges, axis=1) < 1e-6):
        return 100.0
        
    # Compute scores for all vertices
    scores = np.zeros(n)
    for i in range(n):
        rotated = np.roll(polygon, -i, axis=0)
        scores[i] = _compute_vertex_score(rotated)
    
    # If any score is 0, penalize heavily
    if np.any(scores < 0.01):
        return 100.0 + np.sum(1.0 / np.maximum(scores, 1e-6))
        
    # Soft min
    # We want to maximize the minimum score.
    # Let's define the objective as -soft_min(scores)
    # soft_min(scores) = - (1/beta) * log(sum(exp(-beta * scores)))
    # So -soft_min(scores) = (1/beta) * log(sum(exp(-beta * scores)))
    
    beta = 10.0  # Higher beta -> sharper min
    # To avoid overflow, we can subtract max(scores)
    max_score = np.max(scores)
    # If max_score is very large, exp(-beta * (scores - max_score)) is fine.
    # But we are dealing with scores in [0, 1].
    # Let's just compute directly.
    
    # Actually, the scores are between 0 and 1.
    # exp(-beta * scores) will be between exp(-beta) and 1.
    # sum will be between n*exp(-beta) and n.
    # log(sum) is fine.
    
    obj = (1.0 / beta) * np.log(np.sum(np.exp(-beta * scores)))
    
    return obj

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = start_time + 620.0
    
    n = 20
    # Initial guess: regular polygon (all params 0)
    initial_params = np.zeros(20)
    
    best_params = initial_params
    best_score = _polygon_min_score(_params_to_polygon(initial_params))
    best_polygon = _params_to_polygon(initial_params)
    
    # Optimization using Nelder-Mead
    # We use multiple restarts with slightly perturbed initial parameters
    
    rng = np.random.default_rng(42)
    
    num_restarts = 10
    
    for i in range(num_restarts):
        if time.time() > deadline - 60:
            break
            
        # Generate initial parameters
        if i == 0:
            params0 = initial_params
        else:
            # Small random perturbations
            params0 = rng.normal(0, 0.1, 20)
        
        try:
            result = opt.minimize(
                _objective,
                params0,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'maxiter': 200,
                    'adaptive': True
                }
            )
            
            if time.time() > deadline - 30:
                break
                
            candidate_params = result.x
            candidate_polygon = _params_to_polygon(candidate_params)
            candidate_score = _polygon_min_score(candidate_polygon)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_params = candidate_params
                best_polygon = candidate_polygon
                
        except Exception:
            continue
    
    # Final safety check
    final_score = _polygon_min_score(best_polygon)
    if final_score <= 0.0:
        # Fallback to regular polygon
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        
    return best_polygon

# EVOLVE-BLOCK-END
