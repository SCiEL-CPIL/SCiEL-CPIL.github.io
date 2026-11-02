import numpy as np
from scipy.spatial import ConvexHull
import random
import time

N = 13

# EVOLVE-BLOCK-START
def construct():
    n = N
    
    # Precompute triangle indices
    tri_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    n_tris = len(tri_indices)
    
    def get_score(pts):
        """Calculate minimum triangle area / convex hull area"""
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area <= 1e-10:
                return 0.0
        except:
            return 0.0
            
        p1 = pts[tri_indices[:, 0]]
        p2 = pts[tri_indices[:, 1]]
        p3 = pts[tri_indices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        return min_area / hull_area
    
    def compute_areas(pts):
        """Compute all triangle areas (unnormalized)"""
        p1 = pts[tri_indices[:, 0]]
        p2 = pts[tri_indices[:, 1]]
        p3 = pts[tri_indices[:, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)
    
    def get_worst_tri(pts):
        """Return index of worst triangle and its area"""
        areas = compute_areas(pts)
        worst_idx = np.argmin(areas)
        return worst_idx, areas[worst_idx], areas
    
    def local_search(pts, max_iter=200, step_decay=0.95):
        """Iterative local search: move worst triangle points to improve min area"""
        current_score = get_score(pts)
        step_size = 0.05
        no_improve = 0
        
        for iteration in range(max_iter):
            areas = compute_areas(pts)
            worst_idx = np.argmin(areas)
            worst_tri = tri_indices[worst_idx]
            
            best_move_score = current_score
            best_move_pts = None
            
            for pt_idx in worst_tri:
                # Try several random perturbations for this point
                for _ in range(5):
                    new_pts = pts.copy()
                    angle = random.uniform(0, 2 * np.pi)
                    dist = random.uniform(0, step_size)
                    dx = dist * np.cos(angle)
                    dy = dist * np.sin(angle)
                    new_pts[pt_idx, 0] += dx
                    new_pts[pt_idx, 1] += dy
                    
                    new_score = get_score(new_pts)
                    if new_score > best_move_score + 1e-12:
                        best_move_score = new_score
                        best_move_pts = new_pts
            
            if best_move_pts is not None:
                pts = best_move_pts
                current_score = best_move_score
                no_improve = 0
                step_size *= 0.99
            else:
                no_improve += 1
                step_size *= step_decay
                if step_size < 0.001 or no_improve > 30:
                    break
        
        return pts, current_score
    
    def simulated_anneal(pts, start_time, max_time):
        """Simulated annealing with adaptive step size"""
        current_score = get_score(pts)
        best_pts = pts.copy()
        best_score = current_score
        
        T = 0.05
        cooling_rate = 0.9995
        min_T = 1e-4
        step_base = 0.1
        
        while T > min_T and (time.time() - start_time) < max_time:
            new_pts = pts.copy()
            
            # Get current worst triangle
            areas = compute_areas(pts)
            worst_idx = np.argmin(areas)
            worst_tri = tri_indices[worst_idx]
            
            # Choose which point to move
            if random.random() < 0.8:
                # Move a point from the worst triangle
                point_to_move = worst_tri[random.randint(0, 2)]
            else:
                # Move a random point
                point_to_move = random.randint(0, n - 1)
            
            # Perturbation: step size scaled by temperature
            step_size = step_base * (0.1 + 0.9 * T)
            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            
            new_pts[point_to_move, 0] += dx
            new_pts[point_to_move, 1] += dy
            
            new_score = get_score(new_pts)
            
            # Metropolis acceptance
            if new_score >= current_score:
                pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
            else:
                delta = current_score - new_score
                if delta < T:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        pts = new_pts
                        current_score = new_score
            
            T *= cooling_rate
        
        return best_pts, best_score
    
    # Main optimization loop with multiple restarts
    start_time = time.time()
    max_time = 100  # Leave 20s buffer
    
    # Initialize with Fermat spiral
    i = np.arange(n)
    r = np.sqrt(i / n)
    theta = i * 2.399963229728653  # Golden angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.column_stack((x, y))
    points -= np.mean(points, axis=0)
    
    best_points = points.copy()
    best_score = get_score(points)
    
    # Phase 1: Simulated annealing
    sa_pts, sa_score = simulated_anneal(points, start_time, max_time * 0.6)
    if sa_score > best_score:
        best_points = sa_pts
        best_score = sa_score
    
    # Phase 2: Local search on the best solution
    ls_pts, ls_score = local_search(best_points, max_iter=300)
    if ls_score > best_score:
        best_points = ls_pts
        best_score = ls_score
    
    # Phase 3: More SA if time remains
    remaining_time = max_time - (time.time() - start_time)
    if remaining_time > 5:
        sa2_pts, sa2_score = simulated_anneal(best_points, start_time, max_time)
        if sa2_score > best_score:
            best_points = sa2_pts
            best_score = sa2_score
        
        # Final local search
        remaining_time = max_time - (time.time() - start_time)
        if remaining_time > 2:
            ls2_pts, ls2_score = local_search(best_points, max_iter=100)
            if ls2_score > best_score:
                best_points = ls2_pts
                best_score = ls2_score
    
    return best_points
# EVOLVE-BLOCK-END
