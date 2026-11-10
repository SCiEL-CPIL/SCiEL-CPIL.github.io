# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, ConvexHull


def _triangle_area(p1, p2, p3):
    """Compute the area of a triangle given three points."""
    return abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0


def _compute_min_triangle_area(points):
    """Compute the minimum area among all triangles formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


def _compute_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _compute_score(points):
    """Compute the score: A_min / A_hull."""
    a_min = _compute_min_triangle_area(points)
    a_hull = _compute_hull_area(points)
    if a_hull <= 0 or not np.isfinite(a_min):
        return 0.0
    return a_min / a_hull


def _find_min_triangle(points):
    """Find the triangle with minimum area and return its indices and area."""
    n = len(points)
    min_area = float('inf')
    min_idx = (0, 1, 2)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
                    min_idx = (i, j, k)
    return min_idx, min_area


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.
    
    Uses a constrained Delaunay triangulation repair strategy with simulated annealing:
    1. Initialize points on a regular 13-gon.
    2. Iteratively identify the minimum-area triangle among ALL triangles.
    3. Move vertices of that triangle toward centroids of adjacent regions.
    4. Use simulated annealing to escape local minima.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    
    # Initialize on a regular 13-gon (unit circle)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # Simulated annealing parameters
    max_iterations = 3000
    initial_temp = 0.1
    cooling_rate = 0.999
    patience = 100
    no_improve_count = 0
    
    current_points = points.copy()
    current_score = best_score
    temp = initial_temp
    
    for iteration in range(max_iterations):
        # Find the actual minimum-area triangle among all 286 triangles
        min_idx, min_area = _find_min_triangle(current_points)
        
        if min_area <= 1e-10:
            # Points are nearly collinear, reset
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            current_points = np.column_stack([np.cos(angles), np.sin(angles)])
            current_score = _compute_score(current_points)
            continue
        
        # Get the three vertices of the minimum triangle
        i, j, k = min_idx
        
        # For each vertex in the minimum triangle, try multiple move strategies
        improved = False
        
        for vertex_idx in [i, j, k]:
            # Strategy 1: Move toward centroid of all other points
            other_points = np.delete(current_points, vertex_idx, axis=0)
            centroid_all = np.mean(other_points, axis=0)
            
            # Strategy 2: Move toward centroid of Delaunay neighbors
            try:
                tri = Delaunay(current_points)
                neighbor_indices = set()
                for simplex in tri.simplices:
                    if vertex_idx in simplex:
                        for idx in simplex:
                            if idx != vertex_idx:
                                neighbor_indices.add(idx)
                if neighbor_indices:
                    neighbor_points = current_points[list(neighbor_indices)]
                    centroid_neighbors = np.mean(neighbor_points, axis=0)
                else:
                    centroid_neighbors = centroid_all
            except Exception:
                centroid_neighbors = centroid_all
            
            # Strategy 3: Move away from the other two vertices of the min triangle
            other_two = [idx for idx in [i, j, k] if idx != vertex_idx]
            away_dir = current_points[vertex_idx] - np.mean(current_points[other_two], axis=0)
            away_dir_norm = np.linalg.norm(away_dir)
            if away_dir_norm > 1e-10:
                away_dir = away_dir / away_dir_norm
            else:
                away_dir = np.array([1.0, 0.0])
            
            # Generate candidate moves
            candidates = []
            
            # Move toward centroid_all
            for step in [0.02, 0.05, 0.1, 0.15]:
                direction = centroid_all - current_points[vertex_idx]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                    new_point = current_points[vertex_idx] + step * direction
                    candidates.append(new_point)
            
            # Move toward centroid_neighbors
            for step in [0.02, 0.05, 0.1, 0.15]:
                direction = centroid_neighbors - current_points[vertex_idx]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                    new_point = current_points[vertex_idx] + step * direction
                    candidates.append(new_point)
            
            # Move away from other two vertices
            for step in [0.02, 0.05, 0.1, 0.15]:
                new_point = current_points[vertex_idx] + step * away_dir
                candidates.append(new_point)
            
            # Try each candidate
            for new_point in candidates:
                # Check bounds
                if not np.all(np.abs(new_point) <= 2.5):
                    continue
                
                new_points = current_points.copy()
                new_points[vertex_idx] = new_point
                
                # Check convex hull
                try:
                    hull_area = _compute_hull_area(new_points)
                    if hull_area <= 0:
                        continue
                except Exception:
                    continue
                
                new_score = _compute_score(new_points)
                
                if not np.isfinite(new_score):
                    continue
                
                # Simulated annealing acceptance criterion
                if new_score > current_score:
                    # Accept improvement
                    current_points = new_points
                    current_score = new_score
                    
                    if new_score > best_score * (1 + 1e-8):
                        best_points = new_points.copy()
                        best_score = new_score
                        improved = True
                        no_improve_count = 0
                    break
                else:
                    # Accept worse solution with probability exp((new_score - current_score) / temp)
                    delta = new_score - current_score
                    if temp > 1e-10:
                        prob = np.exp(delta / temp)
                        if np.random.random() < prob:
                            current_points = new_points
                            current_score = new_score
                            break
        
        if not improved:
            no_improve_count += 1
        
        # Cool down
        temp *= cooling_rate
        
        if no_improve_count >= patience:
            break
    
    return best_points


# EVOLVE-BLOCK-END
