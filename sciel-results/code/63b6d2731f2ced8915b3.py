"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _covering_radius_fast(points):
    """
    Compute the covering radius in degrees for points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        max_circumradius = 0.0
        for simplex in facets:
            pts = points[simplex]
            n_pts = len(pts)
            if n_pts < 3:
                continue
            
            if n_pts == 3:
                a, b, c = pts[0], pts[1], pts[2]
                ab = np.cross(a, b)
                bc = np.cross(b, c)
                ca = np.cross(c, a)
                triple = np.dot(a, np.cross(b, c))
                denom = 4.0 * abs(triple)
                if denom < 1e-15:
                    continue
                sin_R = (np.linalg.norm(ab) * np.linalg.norm(bc) * np.linalg.norm(ca)) / denom
                sin_R = min(sin_R, 1.0)
                R = np.arcsin(sin_R)
                max_circumradius = max(max_circumradius, R)
            elif n_pts == 4:
                a, b, c, d = pts[0], pts[1], pts[2], pts[3]
                v1 = a - b
                v2 = a - c
                v3 = a - d
                x = np.cross(v1, np.cross(v2, v3))
                nx = np.linalg.norm(x)
                if nx < 1e-15:
                    continue
                x = x / nx
                cos_R = np.clip(np.dot(x, a), -1.0, 1.0)
                R = np.arccos(cos_R)
                max_circumradius = max(max_circumradius, R)
            else:
                centroid = np.mean(pts, axis=0)
                nc = np.linalg.norm(centroid)
                if nc < 1e-15:
                    continue
                centroid = centroid / nc
                for p in pts:
                    cos_R = np.clip(np.dot(centroid, p), -1.0, 1.0)
                    R = np.arccos(cos_R)
                    max_circumradius = max(max_circumradius, R)
        
        return np.degrees(max_circumradius)
    except Exception:
        return np.inf


def _get_worst_facet(points):
    """
    Find the facet with the largest circumradius.
    Returns (max_radius_radians, worst_facet_indices, facet_center)
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        max_circumradius = 0.0
        worst_facet = None
        worst_center = None
        
        for i in range(len(facets)):
            simplex = facets[i]
            pts = points[simplex]
            n_pts = len(pts)
            
            if n_pts < 3:
                continue
            
            R = 0.0
            center = None
            
            if n_pts == 3:
                a, b, c = pts[0], pts[1], pts[2]
                ab = np.cross(a, b)
                bc = np.cross(b, c)
                ca = np.cross(c, a)
                triple = np.dot(a, np.cross(b, c))
                denom = 4.0 * abs(triple)
                if denom < 1e-15:
                    continue
                sin_R = (np.linalg.norm(ab) * np.linalg.norm(bc) * np.linalg.norm(ca)) / denom
                sin_R = min(sin_R, 1.0)
                R = np.arcsin(sin_R)
                v1 = a - b
                v2 = a - c
                x = np.cross(v1, v2)
                nx = np.linalg.norm(x)
                if nx < 1e-15:
                    continue
                x = x / nx
                center = x
            elif n_pts == 4:
                a, b, c, d = pts[0], pts[1], pts[2], pts[3]
                v1 = a - b
                v2 = a - c
                v3 = a - d
                x = np.cross(v1, np.cross(v2, v3))
                nx = np.linalg.norm(x)
                if nx < 1e-15:
                    continue
                x = x / nx
                cos_R = np.clip(np.dot(x, a), -1.0, 1.0)
                R = np.arccos(cos_R)
                center = x
            else:
                centroid = np.mean(pts, axis=0)
                nc = np.linalg.norm(centroid)
                if nc < 1e-15:
                    continue
                centroid = centroid / nc
                center = centroid
                for p in pts:
                    cos_R = np.clip(np.dot(centroid, p), -1.0, 1.0)
                    R = max(R, np.arccos(cos_R))
            
            if R > max_circumradius:
                max_circumradius = R
                worst_facet = simplex
                worst_center = center
        
        return max_circumradius, worst_facet, worst_center
    except Exception:
        return np.inf, None, None


def _optimize_worst_facet(points, max_time=10.0):
    """
    Optimize by moving points that are in the worst facet away from the facet center.
    """
    start_time = time.time()
    best_points = points.copy()
    best_radius = _covering_radius_fast(best_points)
    
    step_size = 0.02
    no_improve_count = 0
    
    while time.time() - start_time < max_time and step_size > 1e-6:
        max_R, worst_facet, worst_center = _get_worst_facet(best_points)
        
        if worst_facet is None or worst_center is None:
            break
        
        improved = False
        for idx in worst_facet:
            if time.time() - start_time > max_time:
                break
            
            p = best_points[idx]
            direction = p - worst_center
            dn = np.linalg.norm(direction)
            if dn < 1e-10:
                continue
            direction = direction / dn
            
            for delta in [step_size, step_size * 0.5, -step_size * 0.3]:
                candidate = best_points.copy()
                candidate[idx] = p + delta * direction
                cn = np.linalg.norm(candidate[idx])
                if cn < 1e-10:
                    continue
                candidate[idx] = candidate[idx] / cn
                
                r = _covering_radius_fast(candidate)
                if r < best_radius - 1e-9:
                    best_points = candidate
                    best_radius = r
                    improved = True
                    break
        
        if not improved:
            step_size *= 0.85
            no_improve_count += 1
            if no_improve_count > 10:
                break
        else:
            no_improve_count = 0
    
    return best_points, best_radius


def _optimize_from_start(points, max_time=15.0):
    """Local optimization combining worst-facet optimization with random perturbations."""
    start_time = time.time()
    best_points = points.copy()
    best_radius = _covering_radius_fast(best_points)
    
    # Phase 1: Worst-facet optimization
    if time.time() - start_time < max_time * 0.6:
        time_left = max_time - (time.time() - start_time)
        opt_points, opt_radius = _optimize_worst_facet(best_points, max_time=time_left * 0.9)
        if opt_radius < best_radius:
            best_points = opt_points
            best_radius = opt_radius
    
    # Phase 2: Random perturbations on all points
    step_size = 0.01
    while time.time() - start_time < max_time:
        improved = False
        n_perturb = max(1, N // 10)
        indices = np.random.choice(N, size=n_perturb, replace=False)
        
        for idx in indices:
            if time.time() - start_time > max_time:
                break
            
            p = best_points[idx]
            random_dir = np.random.randn(3)
            random_dir = random_dir / np.linalg.norm(random_dir)
            
            for delta in [step_size, -step_size, step_size * 0.5]:
                candidate = best_points.copy()
                candidate[idx] = p + delta * random_dir
                cn = np.linalg.norm(candidate[idx])
                if cn < 1e-10:
                    continue
                candidate[idx] = candidate[idx] / cn
                
                r = _covering_radius_fast(candidate)
                if r < best_radius - 1e-9:
                    best_points = candidate
                    best_radius = r
                    improved = True
                    break
        
        if not improved:
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
    return best_points, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points
    best_radius = _covering_radius_fast(best_points)
    
    # Try multiple random restarts with local optimization
    np.random.seed(42)
    
    # Generate diverse starting points
    starts = [best_points]
    
    for restart in range(8):
        rand_points = np.random.randn(N, 3)
        rand_points /= np.linalg.norm(rand_points, axis=1, keepdims=True)
        starts.append(rand_points)
    
    # Antipodal pairs start
    antipodal = np.vstack([np.random.randn(N // 2, 3), -np.random.randn(N // 2, 3)])
    if N % 2 == 1:
        antipodal = np.vstack([antipodal, np.random.randn(1, 3)])
    antipodal /= np.linalg.norm(antipodal, axis=1, keepdims=True)
    starts.append(antipodal)
    
    # Optimize each start
    for start_idx, start_points in enumerate(starts):
        if time.time() > deadline:
            break
        
        time_left = deadline - time.time()
        if time_left < 5.0:
            break
        
        n_remaining = len(starts) - start_idx
        time_per_start = min(15.0, time_left / max(1, n_remaining))
        
        opt_points, opt_radius = _optimize_from_start(start_points, max_time=time_per_start)
        
        if opt_radius < best_radius:
            best_points = opt_points
            best_radius = opt_radius
    
    # Final optimization from the best point found
    if time.time() < deadline - 5.0:
        time_left = deadline - time.time() - 5.0
        final_points, final_radius = _optimize_from_start(best_points, max_time=time_left)
        if final_radius < best_radius:
            best_points = final_points
            best_radius = final_radius
    
    return best_points
# EVOLVE-BLOCK-END
