import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

def compute_metric(points):
    """
    Computes the metric described in the task contract.
    points: (N, 2) array
    Returns: (metric, is_valid)
    """
    N = points.shape[0]
    if N != 20:
        return 0.0, False
    
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0, False
    
    normalized = centered / mean_dist
    
    # Check for repeated points
    min_pair_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(normalized[i] - normalized[j])
            if d < min_pair_dist:
                min_pair_dist = d
    if min_pair_dist < 1e-6:
        return 0.0, False

    # Check strictly convex
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(np.abs(cross) < 1e-12):
        return 0.0, False
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0, False
    
    # Vectorized pairwise distances
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, np.inf)
    
    min_score = np.inf
    for i in range(N):
        d = dist_matrix[i]
        d_sorted = np.sort(d)
        
        if d_sorted[0] < 0.05:
            score = 0.0
        else:
            diffs = np.diff(d_sorted)
            min_diff_idx = np.argmin(diffs)
            k = min_diff_idx
            D = (d_sorted[k] + d_sorted[k+1]) / 2.0
            
            if D < 0.1:
                score = 0.0
            else:
                if k == 0:
                    selected = d_sorted[0:4]
                elif k == 17:
                    selected = d_sorted[15:19]
                else:
                    selected = d_sorted[k-1:k+3]
                
                ratios = np.maximum(selected / D, D / selected)
                raw = np.mean(ratios)
                score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
        
        if score < min_score:
            min_score = score
            
    return min_score, True

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def get_points(params):
        r = np.abs(params[:n])
        off = params[n:]
        # Clamp radii to reasonable range to avoid degenerate polygons
        r = np.clip(r, 0.5, 1.5)
        # Clamp offsets to maintain cyclic order
        off = np.clip(off, -0.15, 0.15)
        
        theta = angles + off
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))
    
    def objective(params):
        if time.time() > deadline:
            return 1.0
        
        points = get_points(params)
        score, valid = compute_metric(points)
        if not valid:
            return 1.0
        return -score
    
    best_score = -1.0
    best_points = None
    
    # 1. Initial guess: Regular polygon
    init_params = np.zeros(2*n)
    init_params[:n] = 1.0
    init_points = get_points(init_params)
    init_score, valid = compute_metric(init_points)
    if valid:
        best_score = init_score
        best_points = init_points.copy()
    
    # 2. Global Search with Differential Evolution
    # Bounds for radii: [0.5, 1.5]
    # Bounds for offsets: [-0.15, 0.15]
    bounds = [(0.5, 1.5)] * n + [(-0.15, 0.15)] * n
    
    try:
        if time.time() < deadline - 60:
            # Use differential_evolution for global exploration
            # maxiter controls the number of generations, popsize controls population size
            # We use a moderate popsize and maxiter to balance exploration and exploitation
            de_res = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=15,
                tol=1e-4,
                init='sobol',
                polish=True  # Let DE polish the result locally
            )
            
            if de_res.success or de_res.fun < 0: # If it found a valid solution
                de_points = get_points(de_res.x)
                de_score, de_valid = compute_metric(de_points)
                if de_valid and de_score > best_score:
                    best_score = de_score
                    best_points = de_points.copy()
                    
    except Exception:
        pass
    
    # 3. Local Refinement with Nelder-Mead
    # If we have a good candidate, refine it further
    if best_points is not None and time.time() < deadline - 20:
        current_params = np.zeros(2*n)
        # Recover params from best_points? 
        # It's easier to just run Nelder-Mead from the best_params found by DE or init
        # Let's keep track of best_params
        
        # Actually, let's just run a few Nelder-Mead restarts from the best point found so far
        # We need to reconstruct params. Since get_points is deterministic, we can't easily invert it.
        # But we stored best_points. We can use Nelder-Mead on the coordinates directly?
        # No, the parameterization is better for convexity.
        
        # Let's assume the best_params from DE were good. 
        # To be safe, let's run Nelder-Mead from the initial regular polygon params and from the DE result if available.
        
        # We didn't store de_res.x in a global var accessible here if it failed.
        # Let's just run a few NM restarts from the regular polygon and some random perturbations.
        
        rng = np.random.default_rng(42)
        
        # Restart 1: From regular polygon
        try:
            if time.time() < deadline - 10:
                nm_res = minimize(
                    objective, 
                    init_params, 
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                nm_points = get_points(nm_res.x)
                nm_score, nm_valid = compute_metric(nm_points)
                if nm_valid and nm_score > best_score:
                    best_score = nm_score
                    best_points = nm_points.copy()
        except Exception:
            pass
        
        # Restart 2-5: From random perturbations of regular polygon
        for i in range(5):
            if time.time() > deadline - 10:
                break
            
            pert_params = init_params.copy()
            pert_params[:n] += rng.normal(0, 0.05, n)
            pert_params[n:] += rng.normal(0, 0.05, n)
            
            try:
                nm_res = minimize(
                    objective, 
                    pert_params, 
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5}
                )
                nm_points = get_points(nm_res.x)
                nm_score, nm_valid = compute_metric(nm_points)
                if nm_valid and nm_score > best_score:
                    best_score = nm_score
                    best_points = nm_points.copy()
            except Exception:
                pass

    if best_points is None:
        return init_points
    
    # Final validation
    final_score, valid = compute_metric(best_points)
    if not valid:
        return init_points
        
    return best_points
