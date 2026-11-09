import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _triangle_areas(points):
    n = len(points)
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    indices = np.array(indices)
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def _score(points):
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        if len(points) != 13 or points.shape != (13, 2):
            return 0.0
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 0:
            return 0.0
        areas = _triangle_areas(points)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0

def heilbronn_convex13() -> np.ndarray:
    n = 13
    deadline = time.time() + 600

    tri_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    tri_indices = tri_indices.T # shape (3, 286)

    def _get_min_tri_info(points):
        areas = _triangle_areas(points)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        i, j, k = tri_indices[:, min_idx]
        return min_area, (i, j, k), areas

    def _local_search(points, max_iters=2000, start_temp=0.1, end_temp=0.001):
        pts = points.copy()
        current_score = _score(pts)
        best_score = current_score
        best_pts = pts.copy()
        
        temp = start_temp
        cooling = (end_temp / start_temp) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline - 30:
                break
                
            min_area, (i, j, k), _ = _get_min_tri_info(pts)
            
            # Try to expand the minimum triangle
            # Move i, j, k slightly away from the centroid of the triangle
            p_i, p_j, p_k = pts[i], pts[j], pts[k]
            centroid = (p_i + p_j + p_k) / 3.0
            
            # Calculate directions to move away from centroid
            dir_i = p_i - centroid
            dir_j = p_j - centroid
            dir_k = p_k - centroid
            
            # Normalize and scale
            norm_i = np.linalg.norm(dir_i)
            norm_j = np.linalg.norm(dir_j)
            norm_k = np.linalg.norm(dir_k)
            
            if norm_i > 1e-10: dir_i /= norm_i
            if norm_j > 1e-10: dir_j /= norm_j
            if norm_k > 1e-10: dir_k /= norm_k
            
            # Step size proportional to temperature and current min area
            step = temp * 0.1
            
            new_pts = pts.copy()
            new_pts[i] += step * dir_i
            new_pts[j] += step * dir_j
            new_pts[k] += step * dir_k
            
            # Add some random noise to escape local minima
            noise = np.random.normal(0, temp * 0.05, size=(n, 2))
            new_pts += noise
            
            new_score = _score(new_pts)
            
            # Accept if better, or with probability exp(-(new_score - current_score)/temp)
            # Note: We want to maximize score, so delta = current - new
            delta = current_score - new_score
            if new_score > current_score or np.random.rand() < np.exp(-delta / temp):
                pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
            
            temp *= cooling
            
        return best_pts, best_score

    # Initializations
    inits = []
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    inits.append(np.column_stack((np.cos(angles), np.sin(angles))))
    
    # Grid-like
    rng = np.random.default_rng(42)
    for s in range(10):
        inits.append(rng.uniform(-1, 1, size=(n, 2)))
    
    best_points = inits[0]
    best_score = _score(best_points)
    
    for init in inits:
        if time.time() > deadline - 60:
            break
        rpts, rscore = _local_search(init, max_iters=1000, start_temp=0.2, end_temp=0.01)
        if rscore > best_score:
            best_score = rscore
            best_points = rpts

    # Final polish with Nelder-Mead on the best point found
    if time.time() < deadline - 30:
        def _obj(x):
            pts = x.reshape(n, 2)
            if np.any(~np.isfinite(pts)):
                return 1e10
            areas = _triangle_areas(pts)
            a_min = np.min(areas)
            if a_min < 1e-15:
                return 1e10
            try:
                hull = ConvexHull(pts)
                a_hull = hull.volume
                if a_hull < 1e-15:
                    return 1e10
                return -(a_min / a_hull)
            except Exception:
                return 1e10

        try:
            res = minimize(_obj, best_points.flatten(), method='nelder-mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            np_ = res.x.reshape(n, 2)
            ns = _score(np_)
            if ns > best_score:
                best_score = ns
                best_points = np_
        except Exception:
            pass

    if _score(best_points) <= 0:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
