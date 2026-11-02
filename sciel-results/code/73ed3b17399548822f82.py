"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


def _covering_radius(points):
    """Compute the covering radius in degrees for a set of points on the unit sphere.
    
    The covering radius is the maximum over all points y on the sphere of the angular
    distance from y to the nearest configuration point. This equals the maximum circumradius
    of the facets of the convex hull, where each facet is checked to have an empty cap.
    """
    points = np.asarray(points)
    n = len(points)
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use a dense grid
        return _covering_radius_grid(points)
    
    # Get facets (simplices in 3D are triangles)
    simplices = hull.simplices
    
    max_circumradius = 0.0
    
    for simplex in simplices:
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Check if the triangle is on the "outer" side (normal pointing outward)
        # The centroid of the triangle
        centroid = (p1 + p2 + p3) / 3.0
        
        # Normal vector
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-12:
            continue
        normal = normal / norm_len
        
        # The outward normal should point away from the origin
        # If the dot product of normal with centroid is negative, flip it
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        # Find the circumcenter of the triangle on the sphere
        # The circumcenter is the point on the sphere equidistant from p1, p2, p3
        # It lies along the normal direction from the origin through the triangle's circumcenter in 3D
        
        # Solve for the circumcenter: find point c such that |c-p1| = |c-p2| = |c-p3| and |c| = 1
        # Using the fact that the circumcenter lies on the line from origin in the direction of normal
        # c = t * normal for some t, but since |c| = 1, c = normal (if the triangle subtends less than 180 degrees)
        
        # Actually, the circumcenter of a spherical triangle is along the normal to the plane of the triangle,
        # but we need to be more careful. Let's use a different approach.
        
        # The covering radius for this facet is the angular distance from the circumcenter to any of the three vertices.
        # The circumcenter of the spherical triangle can be found by intersecting the perpendicular bisectors.
        
        # Alternative: the circumradius is the angle between the circumcenter and a vertex.
        # For a triangle on the unit sphere, if the normal to the plane makes angle alpha with the origin,
        # and the triangle has a circumradius rho in the plane, then the spherical circumradius is...
        
        # Let's just compute it directly: the farthest point on the sphere from the three vertices
        # within the cap defined by the triangle is the circumcenter.
        
        # Compute the circumcenter of the triangle in 3D (the point equidistant from p1, p2, p3)
        # Then project onto the sphere.
        
        # Set up linear system: |c - p1|^2 = |c - p2|^2 = |c - p3|^2
        # This gives: 2*c.(p2-p1) = |p2|^2 - |p1|^2 = 0 (since all on unit sphere)
        # And: 2*c.(p3-p1) = |p3|^2 - |p1|^2 = 0
        # So c is perpendicular to (p2-p1) and (p3-p1), i.e., c is along normal.
        
        # So the circumcenter in 3D is along the normal direction. The point on the sphere
        # closest to all three vertices is normal (or -normal).
        
        # The angular distance from normal to p1:
        cos_angle = np.dot(normal, p1)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Check that this is indeed the circumcenter (equidistant from all three)
        cos_angle2 = np.dot(normal, p2)
        cos_angle3 = np.dot(normal, p3)
        cos_angle2 = np.clip(cos_angle2, -1.0, 1.0)
        cos_angle3 = np.clip(cos_angle3, -1.0, 1.0)
        
        # If the angles are not approximately equal, the circumcenter might be -normal
        if abs(angle - np.arccos(cos_angle2)) > 0.1 or abs(angle - np.arccos(cos_angle3)) > 0.1:
            # Try -normal
            cos_angle = np.dot(-normal, p1)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
        
        max_circumradius = max(max_circumradius, angle)
    
    return np.degrees(max_circumradius)


def _covering_radius_grid(points, n_theta=500, n_phi=500):
    """Fallback covering radius computation using a grid."""
    points = np.asarray(points)
    n = len(points)
    
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    max_dist = 0.0
    for t in theta:
        for p in phi:
            y = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
            dists = np.arccos(np.clip(points @ y, -1.0, 1.0))
            max_dist = max(max_dist, np.min(dists))
    
    return np.degrees(max_dist)


def _objective(params):
    """Objective function: maximize the minimum pairwise angular distance,
    which is equivalent to minimizing the maximum angular separation.
    We use negative log of (1 - max_dot) or similar to make it smooth.
    
    Actually, let's directly minimize the covering radius approximation.
    We'll use the fact that the covering radius is bounded below by the maximum
    circumradius of the Delaunay triangles.
    
    For a smoother objective, we'll use:
    - The maximum over all pairs of the angular distance (to encourage uniform spacing)
    - Plus a penalty for clustering
    
    Actually, a better approach: minimize the sum of squared distances to the target
    minimum distance, or use a soft-min approach.
    
    Let's use: minimize (max_pairwise_distance - target)^2 + small * variance
    But max is not differentiable. Use a soft-max.
    
    Alternative: minimize the covering radius directly via the hull.
    This is non-smooth but we can use subgradient or just use scipy's minimize with Nelder-Mead.
    """
    points = _params_to_points(params)
    
    # Check for coincident points
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            min_dist = min(min_dist, d)
    
    if min_dist < 1e-6:
        return 100.0 + (1e-6 - min_dist)
    
    # Compute covering radius
    cr = _covering_radius(points)
    return cr


def _params_to_points(params):
    """Convert parameters to points on the unit sphere.
    Use spherical coordinates: theta (polar), phi (azimuthal).
    """
    n = len(params) // 2
    theta = params[:n]  # polar angle [0, pi]
    phi = params[n:]    # azimuthal angle [0, 2*pi]
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    points = np.column_stack([x, y, z])
    return points


def _points_to_params(points):
    """Convert points to spherical coordinate parameters."""
    z = points[:, 2]
    x = points[:, 0]
    y = points[:, 1]
    
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    phi = np.mod(phi, 2 * np.pi)
    
    return np.concatenate([theta, phi])


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_score = _covering_radius(best_points)
    
    # Local optimization using Nelder-Mead on the covering radius
    params = _points_to_params(points)
    
    def obj(params):
        points = _params_to_points(params)
        # Check for coincident points
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                min_dist = min(min_dist, d)
        if min_dist < 1e-6:
            return 100.0 + (1e-6 - min_dist)
        return _covering_radius(points)
    
    # Try a few restarts with different initial perturbations
    rng = np.random.default_rng(42)
    
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
        
        if restart == 0:
            current_params = params.copy()
        else:
            # Perturb the initial points
            perturbed = points.copy()
            noise = rng.normal(0, 0.1, perturbed.shape)
            perturbed += noise
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            perturbed = perturbed / norms
            current_params = _points_to_params(perturbed)
        
        try:
            result = minimize(
                obj,
                current_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < best_score:
                new_points = _params_to_points(result.x)
                new_score = _covering_radius(new_points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = new_points.copy()
        except Exception:
            continue
    
    # Final refinement with smaller steps
    if time.time() < deadline - 5.0:
        params = _points_to_params(best_points)
        try:
            result = minimize(
                obj,
                params,
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if result.fun < best_score:
                new_points = _params_to_points(result.x)
                new_score = _covering_radius(new_points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = new_points.copy()
        except Exception:
            pass
    
    # Ensure output is valid
    best_points = np.asarray(best_points)
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END
