"""Rewritten program: spherical potential minimization via L-BFGS-B with multi-restart."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50


def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere.
    
    The covering radius is the maximum, over all points y on the sphere,
    of the angular distance to the nearest point in the configuration.
    This equals the maximum circumradius of the spherical Voronoi cells,
    which for a set of points on the sphere can be computed as:
    - For each point, find its nearest neighbors to determine the Voronoi cell
    - The covering radius is the maximum over all points of the distance to
    the farthest vertex of its Voronoi cell
    
    A simpler approach: the covering radius is the largest d such that there
    exists a point on the sphere at distance d from all configuration points.
    This is equivalent to the maximum over all spherical triangles formed by
    3 points of the circumradius of that triangle (if the triangle contains
    its circumcenter on the sphere), or the maximum distance from a point
    to its nearest neighbor if the Voronoi cell is unbounded in some sense.
    
    For N=50 points on the sphere, the covering radius is determined by the
    largest empty spherical cap. We compute this by:
    1. Computing the convex hull of the points
    2. For each face of the hull, compute the circumradius of the spherical triangle
    3. The covering radius is the max of these circumradii
    """
    from scipy.spatial import ConvexHull
    
    points = np.array(points)
    # Normalize to ensure they're on the unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: if hull fails, use a different method
        return _covering_radius_fallback(points)
    
    max_r = 0.0
    for simplex in hull.simplices:
        # Each simplex is a triangle (3 points) in 3D
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Compute the circumcenter of the spherical triangle
        # The circumcenter is the point on the sphere equidistant from p1, p2, p3
        # and lying on the same side as the face normal
        
        # Normal to the plane through p1, p2, p3
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12:
            continue
        normal = normal / normal_norm
        
        # The circumcenter c satisfies |c - p1| = |c - p2| = |c - p3| = R
        # and c = lambda * normal + offset where offset is in the plane
        
        # For a spherical triangle, the circumcenter is along the normal direction
        # from the centroid of the triangle (in the plane), but we need to be careful.
        
        # Actually, for points on the unit sphere, the circumcenter of the spherical
        # triangle is the unit vector in the direction of the normal, if the triangle
        # is contained in a hemisphere. The circumradius is the angle between the
        # circumcenter and any of the three vertices.
        
        # Let's compute it properly:
        # The circumcenter C (unit vector) satisfies C·p1 = C·p2 = C·p3 = cos(R)
        # where R is the circumradius.
        
        # C = alpha * (p1 x p2) + beta * (p2 x p3) + gamma * (p3 x p1) is one approach,
        # but simpler: C is parallel to normal if the triangle is acute in some sense.
        
        # Let's solve: C = s * normal + t * (p1 + p2 + p3) won't work directly.
        
        # Better: the circumcenter is the intersection of the sphere with the line
        # through the origin in the direction of the normal, IF the normal points
        # in the right direction.
        
        # Check which direction the normal points: it should point outward from the origin
        # toward the face. The face normal from the convex hull already points outward.
        
        # The circumcenter candidate is ±normal. We pick the one that gives a valid
        # circumradius (angle < pi/2 typically for small triangles).
        
        c = normal
        # Check: is c on the same side as the face? The face is part of the convex hull,
        # so the outward normal points away from the interior. The circumcenter of the
        # spherical triangle (which is the farthest point in the Voronoi cell from the
        # three vertices) should be in the direction of the outward normal.
        
        # Compute circumradius: angle between c and p1
        cos_r = np.dot(c, p1)
        cos_r = np.clip(cos_r, -1.0, 1.0)
        r = np.arccos(cos_r)
        
        # Verify it's equidistant
        cos_r2 = np.dot(c, p2)
        cos_r2 = np.clip(cos_r2, -1.0, 1.0)
        r2 = np.arccos(cos_r2)
        
        cos_r3 = np.dot(c, p3)
        cos_r3 = np.clip(cos_r3, -1.0, 1.0)
        r3 = np.arccos(cos_r3)
        
        # If the circumcenter is not valid (i.e., the triangle is obtuse or the
        # circumcenter is on the wrong side), we need to handle it differently.
        # For now, take the max of the three radii as a conservative estimate.
        r_max = max(r, r2, r3)
        
        # But we also need to check that this circumcenter is actually the farthest
        # point in the Voronoi cell. For a valid spherical triangle where the circumcenter
        # lies within the triangle, this is correct.
        
        if r_max > max_r:
            max_r = r_max
    
    # Also check: the covering radius could be limited by points near the "poles"
    # if the distribution is asymmetric. But the convex hull approach should handle this.
    
    return np.degrees(max_r)


def _covering_radius_fallback(points):
    """Fallback covering radius computation using a grid."""
    # Create a fine grid on the sphere
    n_theta = 200
    n_phi = 400
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    # Convert to Cartesian
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Normalize
    norms = np.linalg.norm(grid, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    grid = grid / norms
    
    # Compute minimum distance from each grid point to the configuration
    # Using chunked computation to save memory
    max_min_dist = 0.0
    chunk_size = 10000
    for i in range(0, len(grid), chunk_size):
        chunk = grid[i:i+chunk_size]
        # Compute distances
        # |chunk - points| = sqrt(2 - 2*chunk@points.T)
        dots = chunk @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)  # angular distances
        min_dists = np.min(dists, axis=1)
        max_min_dist = max(max_min_dist, np.max(min_dists))
    
    return np.degrees(max_min_dist)


def _objective(x):
    """Objective function: sum of inverse-power repulsive potentials.
    
    We minimize the sum of 1/r^4 over all pairs, which encourages uniform
    distribution. This is a proxy for minimizing the covering radius.
    """
    points = x.reshape(-1, 3)
    # Compute all pairwise distances
    # r_ij^2 = 2 - 2 * p_i · p_j
    dots = points @ points.T
    np.fill_diagonal(dots, 0.0)
    dots = np.clip(dots, -1.0, 1.0)
    r2 = 2.0 - 2.0 * dots
    # Avoid division by zero for coincident points
    r2 = np.maximum(r2, 1e-12)
    # Potential: sum of 1/r^4 for i < j
    # r^4 = r2^2
    r4 = r2 * r2
    # Only count each pair once
    potential = np.sum(1.0 / r4)
    return potential


def _objective_grad(x):
    """Gradient of the objective function."""
    points = x.reshape(-1, 3)
    n = len(points)
    
    dots = points @ points.T
    np.fill_diagonal(dots, 0.0)
    dots = np.clip(dots, -1.0, 1.0)
    r2 = 2.0 - 2.0 * dots
    r2 = np.maximum(r2, 1e-12)
    r4 = r2 * r2
    
    # d/dp_i (1/r_ij^4) = -4 * (1/r_ij^8) * d(r_ij^2)/dp_i
    # r_ij^2 = 2 - 2*p_i·p_j
    # d(r_ij^2)/dp_i = -2*p_j
    # So d(1/r_ij^4)/dp_i = -4 * (1/r_ij^8) * (-2*p_j) = 8*p_j/r_ij^8
    
    # Sum over all j != i
    # grad_i = sum_{j != i} 8 * p_j / r_ij^8
    
    r8 = r4 * r4
    r8 = np.maximum(r8, 1e-24)
    
    # Weight matrix: W[i,j] = 8 / r_ij^8
    W = 8.0 / r8
    np.fill_diagonal(W, 0.0)
    
    # grad_i = W @ points
    grad = W @ points
    
    return grad.ravel()


def _normalize(points):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return points / norms


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to be safe within 150s limit
    
    best_points = None
    best_score = np.inf
    
    # Multiple restarts with different initial configurations
    rng = np.random.default_rng(42)
    
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Initial configuration
        if restart == 0:
            # Fibonacci spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart == 1:
            # Random points
            points = rng.normal(size=(N, 3))
        elif restart == 2:
            # Fibonacci with perturbation
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            points += 0.1 * rng.normal(size=(N, 3))
        elif restart == 3:
            # Two interleaved Fibonacci spirals
            i1 = np.arange(N // 2) + 0.5
            z1 = 1.0 - 2.0 * i1 / (N // 2)
            phi1 = np.pi * (1.0 + np.sqrt(5.0)) * i1
            r1 = np.sqrt(np.maximum(1.0 - z1 * z1, 0.0))
            pts1 = np.column_stack([r1 * np.cos(phi1), r1 * np.sin(phi1), z1])
            
            i2 = np.arange(N // 2) + 0.5
            z2 = 1.0 - 2.0 * i2 / (N // 2)
            phi2 = np.pi * (1.0 + np.sqrt(5.0)) * i2 + 1.0
            r2 = np.sqrt(np.maximum(1.0 - z2 * z2, 0.0))
            pts2 = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
            
            points = np.vstack([pts1, pts2])
        else:
            # Random with varying scale
            points = rng.normal(size=(N, 3))
            points += 0.5 * rng.normal(size=(N, 3))
        
        points = _normalize(points)
        
        # Optimize using L-BFGS-B
        x0 = points.ravel()
        
        try:
            result = minimize(
                _objective,
                x0,
                jac=_objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            opt_points = _normalize(result.x.reshape(-1, 3))
            
            # Evaluate covering radius
            score = _covering_radius(opt_points)
            
            if score < best_score:
                best_score = score
                best_points = opt_points.copy()
                
        except Exception:
            pass
        
        # If we have time, do a second round of optimization with smaller steps
        if time.time() < deadline - 15.0 and best_points is not None:
            try:
                x1 = best_points.ravel()
                result2 = minimize(
                    _objective,
                    x1,
                    jac=_objective_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'gtol': 1e-12, 'ftol': 1e-14}
                )
                opt_points2 = _normalize(result2.x.reshape(-1, 3))
                score2 = _covering_radius(opt_points2)
                
                if score2 < best_score:
                    best_score = score2
                    best_points = opt_points2.copy()
            except Exception:
                pass
    
    # Final fallback
    if best_points is None:
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return _normalize(best_points)
