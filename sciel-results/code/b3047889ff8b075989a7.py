import numpy as np
import time
import scipy.spatial
import scipy.optimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the smallest triangle area
    to the convex hull area.
    
    We use a physics-based relaxation approach:
    1. Generate diverse initial configurations.
    2. Use force-directed relaxation (Laplacian smoothing + repulsion from small triangles).
    3. Polish the best results using the hard objective with Nelder-Mead.
    4. Return the best configuration found within the time limit.
    """
    n = 13
    start_time = time.time()
    time_limit = 590  # Safety margin before 600s limit

    # Precompute indices for triangles
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    tri_i = i_idx[mask]
    tri_j = j_idx[mask]
    tri_k = k_idx[mask]
    num_tris = len(tri_i)  # 286

    def get_areas(points):
        """Compute areas of all 286 triangles."""
        p_i = points[tri_i]
        p_j = points[tri_j]
        p_k = points[tri_k]
        
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        return areas

    def get_hull_area(points):
        """Compute area of the convex hull."""
        try:
            hull = scipy.spatial.ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def compute_score(points):
        """Compute the hard score: min_triangle_area / hull_area."""
        hull_area = get_hull_area(points)
        if hull_area < 1e-9:
            return 0.0
        areas = get_areas(points)
        min_area = np.min(areas)
        return min_area / hull_area

    def physics_relaxation(points, num_steps=500, alpha=0.1, beta=5.0):
        """
        Physics-based relaxation using Laplacian smoothing and repulsion from small triangles.
        
        Forces:
        1. Laplacian: each point is pulled toward the centroid of its neighbors (promotes even distribution)
        2. Repulsion: points are pushed away from each other, weighted by how small the triangles they form are
        """
        points = points.copy()
        step_size = alpha
        
        for step in range(num_steps):
            if time.time() - start_time > time_limit - 30:
                break
                
            # Compute current areas
            areas = get_areas(points)
            min_area = np.min(areas)
            
            if min_area < 1e-10:
                # Points are nearly collinear, add small random perturbation
                points += np.random.normal(0, 0.01, points.shape)
                continue
            
            # Normalize areas to [0, 1] for weighting
            max_area = np.max(areas)
            if max_area > 1e-10:
                norm_areas = areas / max_area
            else:
                norm_areas = np.ones_like(areas)
            
            # Compute forces
            forces = np.zeros_like(points)
            
            # 1. Laplacian force: pull each point toward centroid of all other points
            for i in range(n):
                others = np.delete(points, i, axis=0)
                centroid = np.mean(others, axis=0)
                forces[i] += (centroid - points[i]) * 0.1
            
            # 2. Repulsion force: push points apart, weighted by small triangle areas
            for t in range(num_tris):
                i, j, k = tri_i[t], tri_j[t], tri_k[t]
                area_t = areas[t]
                
                # Weight is higher for smaller triangles (we want to increase them)
                weight = 1.0 / (area_t + 1e-10)
                
                # Compute triangle centroid
                centroid = (points[i] + points[j] + points[k]) / 3.0
                
                # Push each vertex away from the centroid
                for v in [i, j, k]:
                    direction = points[v] - centroid
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        forces[v] += (direction / norm) * weight * 0.01
            
            # Update points
            points += step_size * forces
            
            # Decay step size
            step_size *= 0.999
        
        return points

    def hard_objective(params):
        """
        Objective function to minimize: - (min_area / hull_area)
        """
        points = params.reshape((n, 2))
        if not np.all(np.isfinite(points)):
            return 1e10
        score = compute_score(points)
        return -score

    # Initial configurations
    rng = np.random.default_rng(seed=42)
    
    # 1. Regular 13-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init1 = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # 2. Random points in a square
    init2 = rng.uniform(-1, 1, (n, 2))
    
    # 3. Random points in a disk
    r = np.sqrt(rng.uniform(0, 1, n))
    theta = rng.uniform(0, 2 * np.pi, n)
    init3 = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    
    # 4. Square grid (4x4 minus 3 points)
    grid = np.array([[x, y] for x in [-1.5, -0.5, 0.5, 1.5] for y in [-1.5, -0.5, 0.5, 1.5]])
    init4 = grid[:13]
    
    # 5. Jittered regular polygon
    init5 = init1 + rng.normal(0, 0.1, (n, 2))
    
    # 6. Jittered grid
    init6 = grid[:13] + rng.normal(0, 0.2, (13, 2))
    
    initializations = [init1, init2, init3, init4, init5, init6]
    
    best_global_params = None
    best_global_score = -1e10
    
    # Run physics-based relaxation for each initialization
    for init in initializations:
        if time.time() - start_time > time_limit - 60:
            break
        
        # Run physics relaxation
        relaxed_points = physics_relaxation(init, num_steps=300, alpha=0.1, beta=5.0)
        
        # Evaluate
        score = compute_score(relaxed_points)
        
        if score > best_global_score:
            best_global_score = score
            best_global_params = relaxed_points.flatten()
    
    # Polish the best result with Nelder-Mead
    if best_global_params is not None and time.time() - start_time < time_limit - 30:
        try:
            res_final = scipy.optimize.minimize(
                hard_objective,
                best_global_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            final_points = res_final.x.reshape((n, 2))
            final_score = compute_score(final_points)
            
            if final_score > best_global_score:
                best_global_score = final_score
                best_global_params = res_final.x.copy()
                
        except Exception:
            pass

    # Final check and return
    if best_global_params is None:
        best_global_params = init1.flatten()
        
    final_points = best_global_params.reshape((n, 2))
    
    if not np.all(np.isfinite(final_points)):
        final_points = init1
        
    return final_points

# EVOLVE-BLOCK-END
