import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import sys

N = 130

def _compute_covering_radius(points):
    """Compute the covering radius (in degrees) for points on unit sphere."""
    try:
        hull = ConvexHull(points)
        max_cos = 1.0
        for simplex in hull.simplices:
            verts = points[simplex]
            # Compute normal to the facet
            v1, v2, v3 = verts[0], verts[1], verts[2]
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            dist = np.abs(np.dot(normal, v1))
            dist = np.clip(dist, -1.0, 1.0)
            cos_angle = dist
            max_cos = min(max_cos, cos_angle)
        
        if max_cos > 1.0:
            max_cos = 1.0
        if max_cos < -1.0:
            max_cos = -1.0
        
        radius_rad = np.arccos(max_cos)
        return np.degrees(radius_rad)
    except Exception:
        return 180.0

def _potential_energy(params):
    """
    Calculate the repulsive potential energy.
    params: array of shape (N, 2) containing theta, phi for each point.
    """
    theta = params[:, 0]
    phi = params[:, 1]
    
    # Convert to Cartesian
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    points = np.column_stack([x, y, z])
    
    # Calculate pairwise squared distances
    # dist2[i, j] = ||points[i] - points[j]||^2
    # Using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i=j
    np.fill_diagonal(dist2, 1.0)
    
    # Potential is sum of 1/dist2 for i < j
    # We can sum all off-diagonal elements and divide by 2
    total = np.sum(1.0 / dist2) - N  # Subtract diagonal terms (1.0)
    return total / 2.0

def _params_to_points(params):
    theta = params[:, 0]
    phi = params[:, 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial configurations
    configs = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Convert to spherical coords
    theta_fib = np.arccos(pts_fib[:, 2])
    phi_fib = np.arctan2(pts_fib[:, 1], pts_fib[:, 0])
    configs.append(np.column_stack([theta_fib, phi_fib]))
    
    # 2. Random
    np.random.seed(42)
    theta_rand = np.random.uniform(0, np.pi, N)
    phi_rand = np.random.uniform(0, 2*np.pi, N)
    configs.append(np.column_stack([theta_rand, phi_rand]))
    
    # 3. Another Random with different seed
    np.random.seed(123)
    theta_rand2 = np.random.uniform(0, np.pi, N)
    phi_rand2 = np.random.uniform(0, 2*np.pi, N)
    configs.append(np.column_stack([theta_rand2, phi_rand2]))
    
    # 4. Grid-like (approximate)
    # Just use a perturbed Fibonacci
    configs.append(configs[0] + np.random.normal(0, 0.1, configs[0].shape))
    
    best_points = None
    best_radius = 180.0
    
    for idx, init_params in enumerate(configs):
        if time.time() > deadline:
            break
            
        # Optimize using L-BFGS-B
        # Note: L-BFGS-B can be slow if the landscape is rough. 
        # We minimize the potential energy.
        res = minimize(_potential_energy, init_params, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.fun < 1e6: # Check for reasonable result
            opt_params = res.x
            points = _params_to_points(opt_params)
            
            # Ensure points are normalized (should be, but just in case)
            norms = np.linalg.norm(points, axis=1)
            points = points / norms[:, np.newaxis]
            
            # Evaluate covering radius
            radius = _compute_covering_radius(points)
            
            if radius < best_radius:
                best_radius = radius
                best_points = points.copy()
                
            print(f"Config {idx}: Radius = {radius:.6f}, Time = {time.time()-start_time:.1f}s", file=sys.stderr)
            
        if time.time() > deadline:
            break
            
    # If we have time, try to refine the best point further
    if best_points is not None and time.time() < deadline - 20:
        # Use Powell method on Cartesian coordinates with a smooth approximation of covering radius
        # This is a structural change: direct optimization of the covering radius proxy
        
        def smooth_covering_radius_cartesian(pts_flat):
            pts = pts_flat.reshape(-1, 3)
            # Normalize
            norms = np.linalg.norm(pts, axis=1)
            norms[norms < 1e-10] = 1.0
            pts = pts / norms[:, np.newaxis]
            
            try:
                hull = ConvexHull(pts)
                cos_vals = []
                for simplex in hull.simplices:
                    verts = pts[simplex]
                    v1, v2, v3 = verts[0], verts[1], verts[2]
                    edge1 = v2 - v1
                    edge2 = v3 - v1
                    normal = np.cross(edge1, edge2)
                    norm = np.linalg.norm(normal)
                    if norm < 1e-12:
                        continue
                    normal = normal / norm
                    dist = np.abs(np.dot(normal, v1))
                    cos_vals.append(dist)
                
                if not cos_vals:
                    return 0.0
                    
                min_cos = min(cos_vals)
                # Angular radius
                rad = np.arccos(np.clip(min_cos, -1.0, 1.0))
                # Smooth approximation: we want to minimize the max radius.
                # The max radius corresponds to the min cos.
                # So we can just return the radius in radians, or a smooth function of it.
                # Since we are using Powell (derivative-free), we can just return the radius.
                return np.degrees(rad)
            except Exception:
                return 180.0

        # Powell is slow. Let's try Nelder-Mead or just a few iterations of a local search.
        # Actually, let's just do a few random perturbations and keep the best.
        current_points = best_points.copy()
        current_radius = best_radius
        
        for step in range(50):
            if time.time() > deadline:
                break
                
            # Perturb points
            new_points = current_points + np.random.normal(0, 0.05, (N, 3))
            norms = np.linalg.norm(new_points, axis=1)
            new_points = new_points / norms[:, np.newaxis]
            
            new_radius = _compute_covering_radius(new_points)
            
            if new_radius < current_radius:
                current_points = new_points
                current_radius = new_radius
            else:
                # Annealing-like: accept with probability exp(-delta/T)
                # T decreases over time
                T = 0.01 * (1 - step/50.0) + 0.001
                if np.random.random() < np.exp(-(new_radius - current_radius) / T):
                    current_points = new_points
                    current_radius = new_radius
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
                
    return best_points
