import numpy as np
from scipy.spatial import ConvexHull
import time

N = 13

# EVOLVE-BLOCK-START
def construct():
    """
    Place 13 points in a planar convex region to maximize the minimum triangle area
    divided by the convex hull area.
    
    Strategy:
    1. Initialize points on concentric circles (exploiting known Heilbronn structure).
    2. Use a local search / gradient-free optimization to improve the configuration.
    3. The objective is: min_triangle_area / convex_hull_area
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Precompute all triangle indices
    tri_indices = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    n_tris = len(tri_indices)
    
    def compute_score(pts):
        """Compute min triangle area / convex hull area."""
        # Compute all triangle areas
        i_idx = tri_indices[:, 0]
        j_idx = tri_indices[:, 1]
        k_idx = tri_indices[:, 2]
        
        p_i = pts[i_idx]
        p_j = pts[j_idx]
        p_k = pts[k_idx]
        
        # Triangle area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        # Convex hull area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume  # In 2D, volume is area
        except Exception:
            return 0.0
        
        if hull_area < 1e-12:
            return 0.0
        
        return min_area / hull_area
    
    def initialize_concentric():
        """Initialize points on concentric circles."""
        pts = np.zeros((N, 2))
        
        # Place points on 3-4 concentric circles
        # Circle 1: 1 point at center
        # Circle 2: 3 points
        # Circle 3: 4 points
        # Circle 4: 5 points
        # Total: 1 + 3 + 4 + 5 = 13
        
        # Center
        pts[0] = [0.0, 0.0]
        
        # Ring 1: radius 0.3, 3 points
        r1 = 0.3
        angles1 = np.linspace(0, 2*np.pi, 3, endpoint=False)
        for idx, angle in enumerate(angles1):
            pts[1 + idx] = [r1 * np.cos(angle), r1 * np.sin(angle)]
        
        # Ring 2: radius 0.6, 4 points
        r2 = 0.6
        angles2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        for idx, angle in enumerate(angles2):
            pts[4 + idx] = [r2 * np.cos(angle), r2 * np.sin(angle)]
        
        # Ring 3: radius 1.0, 5 points
        r3 = 1.0
        angles3 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        for idx, angle in enumerate(angles3):
            pts[8 + idx] = [r3 * np.cos(angle), r3 * np.sin(angle)]
        
        return pts
    
    def initialize_fermat():
        """Initialize points using Fermat spiral."""
        n = N
        i = np.arange(n)
        r = np.sqrt(i / n)
        theta = i * 2.399963229728653  # Golden angle
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts = np.column_stack((x, y))
        pts -= np.mean(pts, axis=0)
        return pts
    
    # Try multiple initializations
    best_pts = None
    best_score = -1.0
    
    initializations = [
        initialize_concentric,
        initialize_fermat,
    ]
    
    # Add some random perturbations to concentric
    rng = np.random.RandomState(42)
    for _ in range(5):
        base = initialize_concentric()
        base += rng.normal(0, 0.05, size=base.shape)
        initializations.append(lambda b=base: b.copy())
    
    for init_func in initializations:
        if time.time() > deadline:
            break
        
        pts = init_func()
        score = compute_score(pts)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Local search optimization
    if best_pts is not None:
        current_pts = best_pts.copy()
        current_score = best_score
        
        # Nelder-Mead style optimization with coordinate descent fallback
        step_size = 0.01
        improvement_threshold = 1e-8
        
        # Use a simple coordinate descent / coordinate perturbation approach
        # Focus on perturbing points that are part of the minimum-area triangles
        
        max_iter = 5000
        no_improve_count = 0
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Find the minimum-area triangle
            i_idx = tri_indices[:, 0]
            j_idx = tri_indices[:, 1]
            k_idx = tri_indices[:, 2]
            
            p_i = current_pts[i_idx]
            p_j = current_pts[j_idx]
            p_k = current_pts[k_idx]
            
            v1 = p_j - p_i
            v2 = p_k - p_i
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            
            min_idx = np.argmin(areas)
            min_tri = tri_indices[min_idx]
            min_area_val = areas[min_idx]
            
            # Try perturbing each point in the minimum triangle
            improved = False
            
            for point_idx in min_tri:
                # Try small perturbations in random directions
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    
                    direction = rng.normal(0, 1, size=2)
                    direction /= (np.linalg.norm(direction) + 1e-12)
                    
                    new_pts = current_pts.copy()
                    new_pts[point_idx] += step_size * direction
                    
                    new_score = compute_score(new_pts)
                    
                    if new_score > current_score + improvement_threshold:
                        current_pts = new_pts
                        current_score = new_score
                        improved = True
                        no_improve_count = 0
                        break
                    elif new_score > current_score:
                        # Accept small improvements
                        current_pts = new_pts
                        current_score = new_score
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        # Try opposite direction
                        new_pts = current_pts.copy()
                        new_pts[point_idx] -= step_size * direction
                        new_score = compute_score(new_pts)
                        if new_score > current_score + improvement_threshold:
                            current_pts = new_pts
                            current_score = new_score
                            improved = True
                            no_improve_count = 0
                            break
                        elif new_score > current_score:
                            current_pts = new_pts
                            current_score = new_score
                            improved = True
                            no_improve_count = 0
                            break
            
            if not improved:
                no_improve_count += 1
                # Reduce step size
                step_size *= 0.9
                if step_size < 1e-6:
                    break
            
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
            
            if no_improve_count > 50:
                break
    
    # Final refinement with smaller steps
    if best_pts is not None and time.time() < deadline:
        current_pts = best_pts.copy()
        current_score = best_score
        step_size = 0.001
        
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            # Find minimum-area triangle
            i_idx = tri_indices[:, 0]
            j_idx = tri_indices[:, 1]
            k_idx = tri_indices[:, 2]
            
            p_i = current_pts[i_idx]
            p_j = current_pts[j_idx]
            p_k = current_pts[k_idx]
            
            v1 = p_j - p_i
            v2 = p_k - p_i
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            
            min_idx = np.argmin(areas)
            min_tri = tri_indices[min_idx]
            
            improved = False
            for point_idx in min_tri:
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    
                    direction = rng.normal(0, 1, size=2)
                    direction /= (np.linalg.norm(direction) + 1e-12)
                    
                    new_pts = current_pts.copy()
                    new_pts[point_idx] += step_size * direction
                    new_score = compute_score(new_pts)
                    
                    if new_score > current_score:
                        current_pts = new_pts
                        current_score = new_score
                        improved = True
                        break
                    else:
                        new_pts = current_pts.copy()
                        new_pts[point_idx] -= step_size * direction
                        new_score = compute_score(new_pts)
                        if new_score > current_score:
                            current_pts = new_pts
                            current_score = new_score
                            improved = True
                            break
            
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
            
            if not improved:
                step_size *= 0.95
                if step_size < 1e-7:
                    break
    
    if best_pts is None:
        best_pts = initialize_concentric()
    
    return best_pts
# EVOLVE-BLOCK-END
