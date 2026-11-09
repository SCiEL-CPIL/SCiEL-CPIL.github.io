import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _get_hull_data(points):
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, np.pi, False
    
    simplices = hull.simplices
    v1 = points[simplices[:, 0]]
    v2 = points[simplices[:, 1]]
    v3 = points[simplices[:, 2]]
    
    n = np.cross(v2 - v1, v3 - v1)
    norms = np.linalg.norm(n, axis=1)
    valid = norms > 1e-10
    if not np.any(valid):
        return simplices, np.array([]), np.pi, False
        
    idx_valid = np.where(valid)[0]
    simplices_valid = simplices[idx_valid]
    
    v1_valid = v1[idx_valid]
    v2_valid = v2[idx_valid]
    v3_valid = v3[idx_valid]
    n_valid = n[idx_valid]
    norms_valid = norms[idx_valid]
    
    c_circ = n_valid / norms_valid[:, None]
    centroids = (v1_valid + v2_valid + v3_valid) / 3.0
    dots = np.sum(c_circ * centroids, axis=1)
    flip = dots < 0
    c_circ[flip] = -c_circ[flip]
    
    cos_theta = np.sum(c_circ * v1_valid, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    max_radius = np.max(radii)
    
    return simplices_valid, radii, max_radius, True

def _radii_for_fixed_hull(points, simplices):
    if len(simplices) == 0:
        return np.array([])
        
    v1 = points[simplices[:, 0]]
    v2 = points[simplices[:, 1]]
    v3 = points[simplices[:, 2]]
    
    n = np.cross(v2 - v1, v3 - v1)
    norms = np.linalg.norm(n, axis=1)
    valid = norms > 1e-10
    
    if not np.any(valid):
        return np.full(len(simplices), np.pi)
        
    radii = np.full(len(simplices), np.pi)
    
    idx_valid = np.where(valid)[0]
    v1_valid = v1[idx_valid]
    n_valid = n[idx_valid]
    norms_valid = norms[idx_valid]
    
    c_circ = n_valid / norms_valid[:, None]
    centroids = (v1[idx_valid] + v2[idx_valid] + v3[idx_valid]) / 3.0
    dots = np.sum(c_circ * centroids, axis=1)
    flip = dots < 0
    c_circ[flip] = -c_circ[flip]
    
    cos_theta = np.sum(c_circ * v1_valid, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii[idx_valid] = np.arccos(cos_theta)
    
    return radii

def _optimize_block(points, start_time, deadline):
    """
    Runs L-BFGS-B optimization with fixed hull topology for a few blocks.
    """
    beta = 20.0
    best_local_points = points.copy()
    best_local_radius, _, _ = _get_hull_data(best_local_points)
    
    max_blocks = 20
    
    for block in range(max_blocks):
        if time.time() > deadline - 10:
            break
            
        simplices, radii, current_radius, valid = _get_hull_data(best_local_points)
        if not valid:
            best_local_points += np.random.randn(N, 3) * 0.01
            best_local_points /= np.linalg.norm(best_local_points, axis=1, keepdims=True)
            continue
            
        def objective(x):
            p = x.reshape(N, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            p = p / norms
            
            r = _radii_for_fixed_hull(p, simplices)
            
            if len(r) == 0:
                return np.pi
            return np.log(np.sum(np.exp(beta * r))) / beta

        x0 = best_local_points.flatten()
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 30, 'gtol': 1e-5})
        except Exception:
            continue
            
        if res.success or res.status == 0:
            new_points = res.x.reshape(N, 3)
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            new_sim, new_rad, new_radius, new_valid = _get_hull_data(new_points)
            
            if new_valid and new_radius < best_local_radius:
                best_local_radius = new_radius
                best_local_points = new_points
                beta = min(beta * 1.1, 50.0)
            else:
                # Perturb to escape
                best_local_points += np.random.randn(N, 3) * 0.005
                best_local_points /= np.linalg.norm(best_local_points, axis=1, keepdims=True)
        else:
            best_local_points += np.random.randn(N, 3) * 0.01
            best_local_points /= np.linalg.norm(best_local_points, axis=1, keepdims=True)

    return best_local_points, best_local_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    np.random.seed(42)
    
    best_points = _fibonacci_sphere(N)
    best_radius, _, _ = _get_hull_data(best_points)
    
    # Optimization Strategy:
    # 1. Optimize from Fibonacci start
    # 2. Optimize from Random start
    # 3. Final Polish
    
    # Start 1: Fibonacci
    if time.time() < deadline - 30:
        pts, rad = _optimize_block(best_points, start_time, deadline)
        if rad < best_radius:
            best_radius = rad
            best_points = pts
            
    # Start 2: Random
    if time.time() < deadline - 30:
        rand_pts = np.random.randn(N, 3)
        rand_pts /= np.linalg.norm(rand_pts, axis=1, keepdims=True)
        # Add some structure to help
        fib_pts = _fibonacci_sphere(N)
        mix_pts = 0.5 * rand_pts + 0.5 * fib_pts
        mix_pts /= np.linalg.norm(mix_pts, axis=1, keepdims=True)
        
        pts, rad = _optimize_block(mix_pts, start_time, deadline)
        if rad < best_radius:
            best_radius = rad
            best_points = pts

    # Final Polish: Greedy worst-facet reduction
    if time.time() < deadline - 5:
        pts = best_points.copy()
        step_size = 0.02
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            try:
                hull = ConvexHull(pts)
                simplices = hull.simplices
                v1 = pts[simplices[:, 0]]
                v2 = pts[simplices[:, 1]]
                v3 = pts[simplices[:, 2]]
                n = np.cross(v2 - v1, v3 - v1)
                norms = np.linalg.norm(n, axis=1)
                valid = norms > 1e-10
                if not np.any(valid): break
                idx_valid = np.where(valid)[0]
                v1_valid = v1[idx_valid]
                v2_valid = v2[idx_valid]
                v3_valid = v3[idx_valid]
                n_valid = n[idx_valid]
                norms_valid = norms[idx_valid]
                c_circ = n_valid / norms_valid[:, None]
                centroids = (v1_valid + v2_valid + v3_valid) / 3.0
                dots = np.sum(c_circ * centroids, axis=1)
                flip = dots < 0
                c_circ[flip] = -c_circ[flip]
                cos_theta = np.sum(c_circ * v1_valid, axis=1)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                radii = np.arccos(cos_theta)
                
                max_idx = np.argmax(radii)
                worst_facet = simplices[idx_valid[max_idx]]
                worst_center = c_circ[max_idx]
                
                v1_idx, v2_idx, v3_idx = worst_facet
                v1, v2, v3 = pts[v1_idx], pts[v2_idx], pts[v3_idx]
                
                alpha = step_size
                p1_new = v1 + alpha * (worst_center - v1)
                p2_new = v2 + alpha * (worst_center - v2)
                p3_new = v3 + alpha * (worst_center - v3)
                
                n1 = np.linalg.norm(p1_new)
                n2 = np.linalg.norm(p2_new)
                n3 = np.linalg.norm(p3_new)
                
                if n1 > 1e-10 and n2 > 1e-10 and n3 > 1e-10:
                    pts[v1_idx] = p1_new / n1
                    pts[v2_idx] = p2_new / n2
                    pts[v3_idx] = p3_new / n3
                
                step_size *= 0.999
                if step_size < 1e-7:
                    break
                    
                new_rad, _, _ = _get_hull_data(pts)
                if new_rad < best_radius:
                    best_radius = new_rad
                    best_points = pts.copy()
            except Exception:
                break

    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    return best_points
