import numpy as np
import time
from scipy.optimize import differential_evolution

def _vertex_score(v, pts):
    """Compute the vertex score for vertex v given all points pts."""
    diffs = pts - v
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    mask = np.ones(len(pts), dtype=bool)
    mask[np.argmin(dists)] = False
    d = np.sort(dists[mask])
    
    # Check for too-close points
    if np.any(d < 0.05):
        return 0.0
        
    gaps = np.diff(d)
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    
    # Check for small gap center
    if D < 0.1:
        return 0.0
        
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k - 1:k + 3]
        
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _min_vertex_score(pts):
    scores = [_vertex_score(pts[i], pts) for i in range(len(pts))]
    return min(scores)

def _is_valid(pts):
    if len(pts) != 20:
        return False
    if not np.all(np.isfinite(pts)):
        return False
    for i in range(20):
        for j in range(i + 1, 20):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-12:
                return False
    signs = []
    for i in range(20):
        p0 = pts[i]
        p1 = pts[(i + 1) % 20]
        p2 = pts[(i + 2) % 20]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-14:
            return False
        signs.append(np.sign(cross))
    if not all(s == signs[0] for s in signs):
        return False
    return True

def _normalize(pts):
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    dists = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return pts_norm, False
    pts_norm = pts_norm / mean_dist
    return pts_norm, True

def _edges_to_points(edge_lengths, angles):
    """
    Reconstruct polygon vertices from edge lengths and turning angles.
    angles: exterior angles (turning angles) between consecutive edges.
    Sum of angles should be 2*pi.
    """
    n = len(edge_lengths)
    # Ensure angles sum to 2*pi? No, we optimize them, but for reconstruction
    # we just use the given angles. The closure is checked by the objective.
    
    # Start at origin
    pts = np.zeros((n + 1, 2))
    current_angle = 0.0
    
    for i in range(n):
        pts[i + 1] = pts[i] + edge_lengths[i] * np.array([np.cos(current_angle), np.sin(current_angle)])
        current_angle += angles[i]
        
    # Check closure implicitly by returning first n points
    return pts[:n]

def _objective(params):
    """
    Objective for Differential Evolution.
    params: 20 edge lengths + 20 angles.
    We want to maximize min_vertex_score.
    DE minimizes, so we return -min_vertex_score + penalty.
    """
    n = 20
    edge_lengths = params[:n]
    angles = params[n:]
    
    # Edge lengths must be positive
    if np.any(edge_lengths <= 0):
        return 1e6
        
    # Angles must be positive and sum to 2*pi for a convex polygon loop?
    # Actually, for a closed convex polygon, sum of exterior angles is 2*pi.
    # If we don't enforce sum=2*pi in the parameterization, the polygon won't close.
    # So we should normalize angles to sum to 2*pi? Or optimize them freely and penalize non-closure?
    # Let's normalize angles to sum to 2*pi to ensure the "turning" is correct, 
    # but the edge lengths determine the shape.
    # Wait, if we normalize angles, we lose the ability to vary the "roundness" relative to closure?
    # No, the shape is determined by the ratio of edge lengths and the specific angles.
    # If we force sum(angles)=2*pi, we are searching in the space of convex polygons.
    
    total_angle = np.sum(angles)
    if total_angle <= 0:
        return 1e6
        
    # Normalize angles to sum to 2*pi
    angles_norm = angles * (2 * np.pi / total_angle)
    
    # Ensure angles are not too small (convexity requires > 0, but for strict convexity and stability)
    if np.any(angles_norm < 1e-3):
        return 1e6
        
    pts = _edges_to_points(edge_lengths, angles_norm)
    
    # Check closure
    closure_error = np.linalg.norm(pts[0] - pts[n])
    if closure_error > 0.1: # Relative tolerance? Let's use absolute for now, scale later?
        # Scale penalty by magnitude of edges to be robust
        scale = np.mean(edge_lengths)
        if closure_error / scale > 0.01:
            return 1e6 + closure_error / scale
            
    norm_pts, valid = _normalize(pts)
    if not valid:
        return 1e6
        
    if not _is_valid(norm_pts):
        return 1e6
        
    score = _min_vertex_score(norm_pts)
    
    # Penalty for non-closure to guide towards closed loops
    penalty = (closure_error / scale) ** 2
    return -score + 100.0 * penalty

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    n = 20
    
    # Initial best: Regular polygon
    # For regular polygon:
    # Edge length L = 2 * R * sin(pi/20)
    # Exterior angle = 2*pi/20
    # We need to find params that generate this.
    # If we set all edge_lengths to 1 and all angles to 2*pi/20, it closes?
    # Sum of vectors: sum_{k=0}^{19} exp(i * k * 2*pi/20) = 0. Yes.
    
    init_edges = np.ones(n)
    init_angles = np.full(n, 2 * np.pi / n)
    init_params = np.concatenate([init_edges, init_angles])
    
    init_pts = _edges_to_points(init_edges, init_angles)
    norm_init, v = _normalize(init_pts)
    
    best_score = 0.0
    best_pts = init_pts.copy()
    
    if v and _is_valid(norm_init):
        best_score = _min_vertex_score(norm_init)
        
    # Define bounds for DE
    # Edge lengths: 0.1 to 10
    # Angles: 0.01 to pi (must be positive for convexity, < pi for strict convexity? 
    # Actually exterior angle can be up to pi for a degenerate triangle-like, but for 20-gon it should be small)
    # Let's allow angles up to pi/2 to be safe, or just 0.01 to 3.0
    bounds = [(0.1, 10.0) for _ in range(n)] + [(0.01, 3.0) for _ in range(n)]
    
    # Run Differential Evolution
    # We have ~10 minutes. DE can be slow.
    # popsize=10, maxiter=500 might be too slow or too fast.
    # Let's try a moderate setting.
    
    try:
        # Use a callback to check time and keep best
        def callback(xk, convergence=True):
            if time.time() > deadline - 30:
                return True
            return False
            
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=15, 
            tol=1e-4, 
            callback=callback,
            polish=False # Polish is local opt, we want global
        )
        
        if res.x is not None:
            trial_pts = _edges_to_points(res.x[:n], res.x[n:])
            # Re-normalize angles for reconstruction if we did so in objective?
            # _edges_to_points uses the angles passed. In objective we normalized them.
            # So we must pass the normalized angles to _edges_to_points for reconstruction?
            # Or just re-run the logic.
            
            # Actually, _objective normalized angles internally. 
            # So res.x contains the *unnormalized* angles that were optimized.
            # We need to reconstruct using the same normalization logic.
            
            angles_opt = res.x[n:]
            total_angle = np.sum(angles_opt)
            if total_angle > 0:
                angles_norm = angles_opt * (2 * np.pi / total_angle)
            else:
                angles_norm = init_angles
                
            trial_pts = _edges_to_points(res.x[:n], angles_norm)
            
            norm_trial, v_trial = _normalize(trial_pts)
            if v_trial and _is_valid(norm_trial):
                trial_score = _min_vertex_score(norm_trial)
                if trial_score > best_score:
                    best_score = trial_score
                    best_pts = trial_pts.copy()
                    
    except Exception as e:
        pass

    # If DE didn't improve much, or failed, try a few random restarts with DE? 
    # Or just return best.
    
    # One more check: if best_score is 0, try a simple random search for a valid polygon?
    if best_score == 0.0:
         # Fallback: Regular polygon is always valid and scores 1.0
         # So best_score should be at least 1.0 from init.
         # If it's 0, something went wrong with init check?
         # Re-verify init
         if v and _is_valid(norm_init):
             best_pts = init_pts.copy()
             best_score = _min_vertex_score(norm_init)
         else:
             # Construct regular polygon manually as fallback
             angles = np.linspace(0, 2*np.pi, n, endpoint=False)
             best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
             
    return best_pts
