import numpy as np
from scipy.optimize import differential_evolution

def _vertex_score(distances):
    d = np.sort(distances)
    if np.any(d < 0.05):
        return 0.0
    
    diffs = np.diff(d)
    k = np.argmin(diffs)
    D = (d[k] + d[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    if k == 0:
        idxs = [0, 1, 2, 3]
    elif k == 17:
        idxs = [15, 16, 17, 18]
    else:
        idxs = [k-1, k, k+1, k+2]
        
    vals = []
    for i in idxs:
        di = d[i]
        ratio = max(di / D, D / di)
        vals.append(ratio)
        
    raw = np.mean(vals)
    return min(1.0, 1.0 / raw)

def _evaluate(points):
    n = len(points)
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    
    dists_to_centroid = np.linalg.norm(pts, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return -100.0
        
    pts = pts / mean_dist
    
    # Check convexity
    cross_prods = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i+1)%n]
        p2 = pts[(i+2)%n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        cross_prods.append(cross)
        
    cross_prods = np.array(cross_prods)
    if np.any(cross_prods <= 0):
        penalty = np.sum(np.abs(cross_prods[cross_prods <= 0]))
        return -100.0 - penalty
        
    # Check min distance
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(pts[i] - pts[j])
            if dist < min_dist:
                min_dist = dist
    if min_dist < 0.05:
        return -100.0
        
    scores = []
    for i in range(n):
        dists = []
        for j in range(n):
            if i != j:
                dists.append(np.linalg.norm(pts[i] - pts[j]))
        score = _vertex_score(np.array(dists))
        scores.append(score)
        
    min_score = np.min(scores)
    return min_score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    bounds = [(-2.0, 2.0)] * (2 * n)
    
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init = np.zeros(2*n)
    for i in range(n):
        init[2*i] = np.cos(angles[i])
        init[2*i+1] = np.sin(angles[i])
        
    def objective(x):
        pts = np.array(x).reshape(n, 2)
        return -_evaluate(pts)
        
    result = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=50, 
        popsize=10,
        tol=0.01,
        init='sobol'
    )
    
    best_points = result.x.reshape(n, 2)
    
    # Normalize as per task
    centroid = np.mean(best_points, axis=0)
    best_points = best_points - centroid
    dists_to_centroid = np.linalg.norm(best_points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist > 0.01:
        best_points = best_points / mean_dist
        
    return best_points
