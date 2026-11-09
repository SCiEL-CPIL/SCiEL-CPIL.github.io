# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, ConvexHull

def _triangle_areas_vectorized(points):
    """
    Compute areas of all triangles formed by triples of points.
    Returns a list of (area, i, j, k) or just the min area and indices.
    For N=13, this is fast.
    """
    n = len(points)
    min_area = float('inf')
    min_idx = None
    
    # Precompute cross products or use direct formula
    # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                
                # Vectorized cross product magnitude for triangle area
                # (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
                cross = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                area = 0.5 * abs(cross)
                
                if area < min_area:
                    min_area = area
                    min_idx = (i, j, k)
                    
    return min_area, min_idx

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    a_min, _ = _triangle_areas_vectorized(points)
    a_hull = _hull_area(points)
    if a_hull <= 0 or not np.isfinite(a_min):
        return 0.0
    return a_min / a_hull

def heilbronn_convex13() -> np.ndarray:
    n = 13
    # Initialize on regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_points = points.copy()
    best_min_area, _ = _triangle_areas_vectorized(points)
    best_hull_area = _hull_area(points)
    best_score = best_min_area / best_hull_area if best_hull_area > 0 else 0.0
    
    max_iterations = 5000
    step_size = 0.1
    patience = 100
    no_improve = 0
    
    # Get Delaunay triangulation for neighbor info
    try:
        tri = Delaunay(points)
    except Exception:
        return best_points

    for it in range(max_iterations):
        # Update Delaunay if points changed significantly? 
        # For local moves, it's expensive to recompute every time.
        # We'll recompute only if we accept a move.
        
        current_min_area, min_idx = _triangle_areas_vectorized(points)
        
        if min_idx is None:
            break
            
        improved = False
        i, j, k = min_idx
        
        # Candidates for each vertex in the minimal triangle
        for v_idx in [i, j, k]:
            # Find neighbors in current Delaunay
            neighbors = []
            for simplex in tri.simplices:
                if v_idx in simplex:
                    for p in simplex:
                        if p != v_idx:
                            neighbors.append(p)
            
            if not neighbors:
                continue
            
            # Centroid of neighbors
            centroid = np.mean(points[neighbors], axis=0)
            
            # Generate candidate positions
            candidates = [
                centroid,
                points[v_idx] + 0.5 * (centroid - points[v_idx]),
                points[v_idx] + 0.2 * (centroid - points[v_idx]),
                points[v_idx] + 0.8 * (centroid - points[v_idx]),
                # Small random perturbations
                points[v_idx] + 0.05 * np.random.normal(size=2),
                points[v_idx] - 0.05 * np.random.normal(size=2),
            ]
            
            for new_pos in candidates:
                # Bounds check
                if not np.all(np.abs(new_pos) < 5):
                    continue
                
                new_points = points.copy()
                new_points[v_idx] = new_pos
                
                # Check hull validity
                new_hull_area = _hull_area(new_points)
                if new_hull_area <= 0:
                    continue
                
                new_min_area, _ = _triangle_areas_vectorized(new_points)
                new_score = new_min_area / new_hull_area
                
                if new_score > best_score * (1 + 1e-9):
                    best_points = new_points.copy()
                    best_min_area = new_min_area
                    best_hull_area = new_hull_area
                    best_score = new_score
                    points = new_points.copy()
                    improved = True
                    no_improve = 0
                    
                    # Update Delaunay
                    try:
                        tri = Delaunay(points)
                    except Exception:
                        break
                    break
            
            if improved:
                break
        
        if not improved:
            no_improve += 1
            # Decay step size / perturbations
            if no_improve > patience:
                break
            if no_improve % 10 == 0:
                # Perturb slightly to escape local minima
                points = best_points + 0.01 * np.random.normal(size=(n, 2))
                try:
                    tri = Delaunay(points)
                except Exception:
                    break

    return best_points
# EVOLVE-BLOCK-END
