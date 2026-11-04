"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
from scipy.spatial import ConvexHull

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points using vectorized ops."""
    n = len(points)
    # Compute all cross products for all triples
    min_area = float('inf')
    # Vectorized approach: compute areas for all triples
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    # Only keep valid triples i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    p1 = points[i_idx]
    p2 = points[j_idx]
    p3 = points[k_idx]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: simple method using shoelace on sorted points
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        sorted_points = points[np.argsort(angles)]
        area = 0.0
        n = len(sorted_points)
        for i in range(n):
            j = (i + 1) % n
            area += sorted_points[i][0] * sorted_points[j][1]
            area -= sorted_points[j][0] * sorted_points[i][1]
        return abs(area) / 2.0

def _objective(x):
    """Objective to minimize: negative of (min_triangle_area / hull_area)."""
    points = x.reshape(N, 2)
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return -min_area / hull_area

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _local_refine(points, iterations=100):
    """Perform local refinement by perturbing the worst-performing triangles."""
    best_points = points.copy()
    best_score = _score(best_points)
    
    for _ in range(iterations):
        # Find the triple with minimum area
        n = len(best_points)
        min_area = float('inf')
        worst_triple = None
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1 = best_points[i]
                    p2 = best_points[j]
                    p3 = best_points[k]
                    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                    area = 0.5 * abs(cross)
                    if area < min_area:
                        min_area = area
                        worst_triple = (i, j, k)
        
        if worst_triple is None:
            break
        
        i, j, k = worst_triple
        # Perturb these three points slightly to increase the minimum area
        for _ in range(10):
            perturbation = np.random.randn(N, 2) * 0.01
            new_points = best_points + perturbation
            new_score = _score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
                break
    
    return best_points, best_score

def construct():
    best_points = None
    best_score = 0.0
    
    # Generate multiple diverse starting configurations
    initial_configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Regular 13-gon with slight random perturbation
    np.random.seed(0)
    angles2 = angles + np.random.randn(N) * 0.1
    initial_configs.append(np.column_stack([np.cos(angles2), np.sin(angles2)]))
    
    # 3. Two concentric circles (7 outer, 6 inner)
    outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_pts = np.column_stack([0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)])
    initial_configs.append(np.vstack([outer_pts, inner_pts]))
    
    # 4. Three rings (5 outer, 4 middle, 4 inner)
    ring1_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    ring2_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    ring3_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 8
    ring1_pts = np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
    ring2_pts = np.column_stack([0.6 * np.cos(ring2_angles), 0.6 * np.sin(ring2_angles)])
    ring3_pts = np.column_stack([0.3 * np.cos(ring3_angles), 0.3 * np.sin(ring3_angles)])
    initial_configs.append(np.vstack([ring1_pts, ring2_pts, ring3_pts]))
    
    # 5. Random points in unit circle
    np.random.seed(42)
    for _ in range(3):
        r = np.sqrt(np.random.rand(N))
        theta = 2 * np.pi * np.random.rand(N)
        initial_configs.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    # Optimize each configuration
    for idx, init_pts in enumerate(initial_configs):
        x0 = init_pts.flatten()
        
        # First pass: Nelder-Mead with moderate iterations
        result = minimize(
            _objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        
        current_points = result.x.reshape(N, 2)
        current_score = _score(current_points)
        
        # Second pass: local refinement
        refined_points, refined_score = _local_refine(current_points, iterations=50)
        
        if refined_score > best_score:
            best_points = refined_points
            best_score = refined_score
    
    # Final aggressive optimization on the best solution found
    if best_points is not None:
        x_final = best_points.flatten()
        result_final = minimize(
            _objective,
            x_final,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        final_points = result_final.x.reshape(N, 2)
        final_score = _score(final_points)
        
        if final_score > best_score:
            best_points = final_points
            best_score = final_score
        
        # One more local refinement
        refined_points, refined_score = _local_refine(best_points, iterations=100)
        if refined_score > best_score:
            best_points = refined_points
            best_score = refined_score
    
    return best_points
# EVOLVE-BLOCK-END
