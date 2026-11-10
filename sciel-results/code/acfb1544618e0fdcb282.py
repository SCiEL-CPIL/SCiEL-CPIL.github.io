import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _covering_radius(points):
    """
    Calculate the covering radius in radians for points on the unit sphere.
    The covering radius is the maximum circumradius of the convex hull facets.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi # Fallback for degenerate cases
        
    max_radius = 0.0
    # hull.simplices contains the indices of the vertices of each facet
    for simplex in hull.simplices:
        p0 = points[simplex[0]]
        p1 = points[simplex[1]]
        p2 = points[simplex[2]]
        
        # Normal to the plane of the triangle
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_normal = np.linalg.norm(normal)
        
        if norm_normal < 1e-10:
            continue
            
        # Distance from origin to the plane
        # d = |normal . p0| / ||normal||
        d = np.abs(np.dot(normal, p0)) / norm_normal
        
        # The circumradius (angular) alpha satisfies cos(alpha) = d
        # Since d is in [0, 1], alpha is in [0, pi/2] for valid hull facets on unit sphere
        # Actually, for a triangle on the unit sphere, the circumcenter is on the sphere
        # and the distance from origin to the plane is cos(alpha).
        # So alpha = arccos(d).
        
        # Clip d to [0, 1] to avoid numerical errors in arccos
        d = np.clip(d, 0.0, 1.0)
        alpha = np.arccos(d)
        
        if alpha > max_radius:
            max_radius = alpha
            
    return max_radius

def _repulsive_relaxation(points, steps=2000, alpha=0.1):
    """
    Relax points using repulsive forces to spread them out.
    """
    points = points.copy()
    for _ in range(steps):
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    continue
                # Force is proportional to 1/dist^2 (Coulomb-like) or 1/dist
                # Using 1/dist^2 for stronger repulsion at close range
                force = diff / (dist_sq * np.sqrt(dist_sq)) # diff / dist^3 ? No, diff is vector of length dist. diff/dist^2 is force direction * 1/dist.
                # Let's use standard repulsion: F ~ r / r^3 = r_hat / r^2.
                # diff is r vector. diff / r^3 = r_hat / r^2.
                forces[i] += diff / (dist_sq * np.sqrt(dist_sq))
        
        # Update positions
        points += alpha * forces
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points /= norms
        
    return points

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # 2. Repulsive Relaxation
    # This is fast, so we can do a lot of steps
    points = _repulsive_relaxation(points, steps=5000, alpha=0.05)
    current_radius = _covering_radius(points)
    if current_radius < best_radius:
        best_points = points.copy()
        best_radius = current_radius
        
    # 3. Local Optimization with Nelder-Mead
    # We optimize the Cartesian coordinates.
    # To handle the sphere constraint, we can optimize on the sphere directly?
    # Nelder-Mead doesn't support constraints well.
    # We can use a trick: optimize on the tangent plane or just use Cartesian and normalize in the objective?
    # If we normalize in the objective, the gradient is broken, but Nelder-Mead is derivative-free.
    # However, the search space is unbounded in Cartesian.
    # Better: Use spherical coordinates (theta, phi) as parameters.
    # theta in [0, pi], phi in [0, 2*pi].
    # This maps to the sphere.
    
    def spherical_to_cartesian(params):
        theta = params[0::2]
        phi = params[1::2]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])
        
    def objective(params):
        # Check time
        if time.time() > deadline:
            return 1e6 # Large penalty
        
        pts = spherical_to_cartesian(params)
        # Ensure points are well-behaved
        if not np.all(np.isfinite(pts)):
            return 1e6
            
        rad = _covering_radius(pts)
        return rad
        
    # Initialize params from current best points
    theta = np.arccos(np.clip(best_points[:, 2], -1.0, 1.0))
    phi = np.arctan2(best_points[:, 1], best_points[:, 0])
    params0 = np.zeros(2 * N)
    params0[0::2] = theta
    params0[1::2] = phi
    
    from scipy.optimize import minimize
    
    # Run Nelder-Mead
    # maxiter needs to be set based on time, but we have the time check in objective.
    # We can also wrap the minimize call in a try-except or check time after.
    
    try:
        result = minimize(objective, params0, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if result.fun < best_radius:
            best_params = result.x
            best_points = spherical_to_cartesian(best_params)
            best_radius = result.fun
            
    except Exception:
        pass
        
    # Check time again
    if time.time() > deadline:
        return best_points
        
    # 4. Maybe one more round of repulsive relaxation + optimization?
    # If we have time left, try a few random restarts or perturbations.
    
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        # Try 5 random perturbations
        for _ in range(5):
            if time.time() > deadline:
                break
                
            # Perturb current best
            noise = np.random.normal(0, 0.1, size=best_points.shape)
            perturbed = best_points + noise
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            perturbed /= norms
            
            # Quick relaxation
            relaxed = _repulsive_relaxation(perturbed, steps=1000, alpha=0.05)
            rad = _covering_radius(relaxed)
            
            if rad < best_radius:
                best_points = relaxed
                best_radius = rad
                
                # Re-optimize from this new start
                theta = np.arccos(np.clip(best_points[:, 2], -1.0, 1.0))
                phi = np.arctan2(best_points[:, 1], best_points[:, 0])
                params0 = np.zeros(2 * N)
                params0[0::2] = theta
                params0[1::2] = phi
                
                try:
                    result = minimize(objective, params0, method='Nelder-Mead', 
                                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
                    if result.fun < best_radius:
                        best_points = spherical_to_cartesian(result.x)
                        best_radius = result.fun
                except Exception:
                    pass

    return best_points
