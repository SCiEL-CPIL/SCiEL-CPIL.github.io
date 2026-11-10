import numpy as np
from scipy.spatial import ConvexHull

def heilbronn_convex13() -> np.ndarray:
    n = 13
    
    def get_score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        try:
            hull = ConvexHull(pts)
            a_hull = hull.volume
        except Exception:
            return 0.0
        if a_hull < 1e-10:
            return 0.0
        
        min_area = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                    if area < min_area:
                        min_area = area
        if min_area < 1e-10:
            return 0.0
        return min_area / a_hull

    def anneal(points, iterations=5000, t0=0.5, t_min=1e-5):
        points = points.copy()
        current_score = get_score(points)
        best_score = current_score
        best_points = points.copy()
        
        for i in range(iterations):
            temp = t0 * (t_min / t0) ** (i / iterations)
            
            idx = np.random.randint(n)
            orig = points[idx].copy()
            points[idx] += np.random.normal(0, 0.1, 2)
            
            new_score = get_score(points)
            
            if new_score > current_score:
                accept = True
            else:
                delta = new_score - current_score
                if delta < -1e-10:
                    p = np.exp(delta / temp)
                else:
                    p = 1.0
                accept = (np.random.random() < p)
            
            if accept:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = points.copy()
            else:
                points[idx] = orig
                
        return best_points, best_score

    rng = np.random.default_rng(seed=42)
    np.random.seed(42) # For np.random in anneal
    
    best_overall = 0.0
    best_pts = None
    
    # Start 1: Regular Polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    start1 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Start 2: Jittered Regular Polygon
    angles2 = angles + rng.uniform(-0.2, 0.2, n)
    radii2 = 1.0 + rng.uniform(-0.2, 0.2, n)
    start2 = np.column_stack([radii2 * np.cos(angles2), radii2 * np.sin(angles2)])
    
    # Start 3: Random in Unit Disk
    start3 = rng.uniform(-1, 1, (n, 2))
    
    starts = [start1, start2, start3]
    
    for s in starts:
        pts, score = anneal(s, iterations=5000, t0=0.5, t_min=1e-5)
        if score > best_overall:
            best_overall = score
            best_pts = pts

    return best_pts
