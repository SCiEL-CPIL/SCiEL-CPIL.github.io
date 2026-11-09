import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import random

N = 130

def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius(points):
    """Compute the covering radius using the convex hull method."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_radius = 0.0
    simplices = hull.simplices
    
    for simplex in simplices:
        v1 = points[simplex[0]]
        v2 = points[simplex[1]]
        v3 = points[simplex[2]]
        
        # Compute circumcenter on the sphere
        # c is equidistant from v1, v2, v3
        # c . v1 = c . v2 = c . v3
        # c is orthogonal to (v1-v2) and (v1-v3)
        n1 = v1 - v2
        n2 = v1 - v3
        cross = np.cross(n1, n2)
        norm = np.linalg.norm(cross)
        if norm < 1e-10:
            continue
        c = cross / norm
        
        # Ensure c is on the correct side
        if c @ v1 < 0:
            c = -c
        
        cos_R = np.clip(c @ v1, -1.0, 1.0)
        R = np.arccos(cos_R)
        
        # Check if circumcenter is inside the spherical triangle
        # by checking if it can be expressed as positive combination
        # Simpler: check angles or use barycentric-like coordinates
        # For now, take max of circumcenter distance and edge midpoints
        
        # Edge midpoints on sphere
        m12 = v1 + v2
        n12 = np.linalg.norm(m12)
        if n12 > 1e-10:
            m12 /= n12
            d12 = np.arccos(np.clip(m12 @ v1, -1.0, 1.0))
        else:
            d12 = np.pi
        
        m23 = v2 + v3
        n23 = np.linalg.norm(m23)
        if n23 > 1e-10:
            m23 /= n23
            d23 = np.arccos(np.clip(m23 @ v2, -1.0, 1.0))
        else:
            d23 = np.pi
        
        m31 = v3 + v1
        n31 = np.linalg.norm(m31)
        if n31 > 1e-10:
            m31 /= n31
            d31 = np.arccos(np.clip(m31 @ v3, -1.0, 1.0))
        else:
            d31 = np.pi
        
        local_r = max(R, d12, d23, d31)
        if local_r > max_radius:
            max_radius = local_r
            
    return max_radius

def _repulsion_energy(points):
    """Compute a surrogate for covering radius using repulsion."""
    # Sum of inverse squared distances to nearest neighbors
    dots = points @ points.T
    np.fill_diagonal(dots, -2.0)  # Avoid self-distance
    nearest_dots = np.max(dots, axis=1)
    nearest_dots = np.clip(nearest_dots, -1.0, 1.0)
    nearest_dists = 1.0 - nearest_dots  # Approximate squared chord distance
    nearest_dists = np.maximum(nearest_dists, 1e-10)
    return np.sum(1.0 / nearest_dists)

def _optimize_single(points, deadline):
    """Run optimization on a single starting configuration."""
    start_time = time.time()
    current_points = points.copy()
    best_points = current_points.copy()
    best_radius = _covering_radius(current_points)
    
    # Phase 1: Lloyd's iterations
    no_improve = 0
    max_lloyd_iters = 80
    for i in range(max_lloyd_iters):
        if time.time() > deadline - 20.0:
            break
        
        # Lloyd step: move each point to spherical mean of its Voronoi cell
        dots = current_points @ current_points.T
        nearest_indices = np.argmax(dots, axis=1)
        
        new_points = np.zeros_like(current_points)
        for j in range(N):
            neighbors = np.where(nearest_indices == j)[0]
            if len(neighbors) > 0:
                cell_points = current_points[neighbors]
                mean_vec = np.sum(cell_points, axis=0)
                norm = np.linalg.norm(mean_vec)
                if norm > 1e-10:
                    new_points[j] = mean_vec / norm
                else:
                    new_points[j] = current_points[j]
            else:
                new_points[j] = current_points[j]
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_points = new_points / norms
        
        new_radius = _covering_radius(new_points)
        if new_radius < best_radius - 1e-10:
            best_points = new_points.copy()
            best_radius = new_radius
            no_improve = 0
        else:
            no_improve += 1
        
        current_points = new_points
        
        if no_improve > 30:
            # Add noise to escape
            noise = np.random.normal(0, 0.02, size=current_points.shape)
            perturbed = current_points + noise
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            current_points = perturbed / norms
            no_improve = 0
    
    # Phase 2: Gradient-based polish using L-BFGS-B
    if time.time() < deadline - 5.0:
        def objective(params):
            p = params.reshape(N, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            p = p / norms
            return _covering_radius(p)
        
        x0 = best_points.flatten()
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 50, 'maxfun': 1000, 'disp': False})
            if res.success:
                p_opt = res.x.reshape(N, 3)
                norms = np.linalg.norm(p_opt, axis=1, keepdims=True)
                p_opt = p_opt / norms
                r_opt = _covering_radius(p_opt)
                if r_opt < best_radius:
                    best_points = p_opt
                    best_radius = r_opt
        except Exception:
            pass
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_points = None
    best_radius = np.pi
    
    # Multiple restarts with different initializations
    random.seed(42)
    
    # Start with Fibonacci
    inits = [fibonacci_sphere(N)]
    
    # Add random initializations
    for _ in range(4):
        angles_theta = np.random.uniform(0, np.pi, N)
        angles_phi = np.random.uniform(0, 2*np.pi, N)
        x = np.sin(angles_theta) * np.cos(angles_phi)
        y = np.sin(angles_theta) * np.sin(angles_phi)
        z = np.cos(angles_theta)
        inits.append(np.column_stack([x, y, z]))
    
    # Optimize each initialization
    for init in inits:
        if time.time() > deadline - 25.0:
            break
        
        pts, rad = _optimize_single(init, deadline)
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
    
    if best_points is None:
        best_points = fibonacci_sphere(N)
    
    return best_points
