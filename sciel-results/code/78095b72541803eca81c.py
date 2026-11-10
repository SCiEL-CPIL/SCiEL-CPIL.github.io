import numpy as np
from scipy.optimize import minimize

def calculate_score(points):
    """
    Calculate the score for a given set of points.
    Returns the minimum vertex score.
    """
    n = len(points)
    pts = np.array(points)
    
    # Check convexity (turning angles)
    # For a strictly convex polygon in CCW order, cross product of consecutive edges should be positive
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i+1)%n]
        p2 = pts[(i+2)%n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if cross <= 0:
            return 0.0

    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    dists_to_centroid = np.linalg.norm(pts, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    pts = pts / mean_dist
    
    # Calculate score for each vertex
    min_score = 1.0
    for i in range(n):
        vi = pts[i]
        # Distances to other vertices
        dists_i = np.linalg.norm(pts - vi, axis=1)
        # Remove self distance (index i)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        dists_i = dists_i[mask]
        
        # Check min distance
        if np.min(dists_i) < 0.05:
            return 0.0
        
        # Sort
        sorted_dists = np.sort(dists_i)
        
        # Find min gap
        gaps = np.diff(sorted_dists)
        k = np.argmin(gaps)
        D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
        
        if D < 0.1:
            return 0.0
            
        # Select 4 distances
        if k == 0:
            idxs = [0, 1, 2, 3]
        elif k == 17:
            idxs = [15, 16, 17, 18]
        else:
            idxs = [k-1, k, k+1, k+2]
            
        selected_dists = sorted_dists[idxs]
        
        # Raw score
        ratios = np.maximum(selected_dists / D, D / selected_dists)
        raw = np.mean(ratios)
        score = min(1.0, 1.0 / raw)
        
        if score < min_score:
            min_score = score
            
    return min_score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    
    # The regular 20-gon is already a very strong candidate.
    # The score depends on the distribution of distances from each vertex to others.
    # For a regular polygon, the distances are symmetrically distributed.
    # Let's check if we can improve by slightly perturbing the angles.
    
    # Initial guess: regular polygon
    angles0 = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # We want to maximize the minimum vertex score.
    # Let's use a local optimization on the angles.
    # Variables: angles (19 free, one fixed to 0 for rotational invariance, or all 20 with constraint sum=2pi)
    # Let's fix angle[0] = 0 and optimize the other 19 angles in (0, 2pi) sorted.
    # Or optimize the gaps.
    
    # Let's try optimizing the gaps.
    # gaps: array of 20 positive numbers summing to 2pi.
    # Use simplex method or SLSQP.
    
    def score_from_gaps(gaps):
        # Normalize gaps to sum to 2pi
        gaps = np.array(gaps)
        if np.any(gaps <= 0):
            return 0.0
        total = np.sum(gaps)
        if total <= 0:
            return 0.0
        gaps = gaps / total * 2 * np.pi
        
        angles = np.cumsum(gaps)
        angles = np.roll(angles, -1)  # angles[i] is the angle of vertex i
        # Actually, let's define vertex i at angle sum(gaps[0..i-1])
        # So vertex 0 is at 0.
        angles = np.zeros(n)
        for i in range(1, n):
            angles[i] = np.sum(gaps[:i])
        
        x = np.cos(angles)
        y = np.sin(angles)
        pts = np.stack([x, y], axis=1)
        
        return calculate_score(pts)
    
    # Initial gaps: all equal
    gaps0 = np.full(n, 2.0 * np.pi / n)
    best_score = score_from_gaps(gaps0)
    best_gaps = gaps0.copy()
    
    # Try to improve with local search
    # Use SLSQP to maximize score_from_gaps
    # Objective: -score_from_gaps
    
    def objective(gaps):
        return -score_from_gaps(gaps)
    
    # Constraints: sum(gaps) = 2pi, gaps > 0
    # SLSQP can handle equality constraints and bounds.
    
    # To avoid numerical issues, we can parameterize gaps as exp(x) and optimize x?
    # Or just use SLSQP with bounds.
    
    # Let's try a few random restarts with small perturbations.
    
    rng = np.random.default_rng(42)
    
    for restart in range(10):
        # Perturb the initial gaps
        if restart == 0:
            gaps_init = gaps0.copy()
        else:
            # Random perturbation
            gaps_init = gaps0 * (1 + 0.1 * rng.standard_normal(n))
            gaps_init = np.abs(gaps_init) + 1e-6 # Ensure positive
        
        # Optimize
        try:
            res = minimize(objective, gaps_init, method='SLSQP', 
                          bounds=[(1e-4, 2*np.pi)]*n,
                          constraints=[{'type': 'eq', 'fun': lambda g: np.sum(g) - 2*np.pi}],
                          options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_gaps = res.x.copy()
        except Exception:
            pass
    
    # Construct the final polygon from best_gaps
    angles = np.zeros(n)
    for i in range(1, n):
        angles[i] = np.sum(best_gaps[:i])
        
    x = np.cos(angles)
    y = np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    return polygon
