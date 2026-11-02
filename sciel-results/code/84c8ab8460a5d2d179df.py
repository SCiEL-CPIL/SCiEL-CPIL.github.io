"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize, differential_evolution
import time


def _covering_radius_exact(points):
    """
    Compute the exact covering radius of points on the unit sphere.
    
    The covering radius is the maximum over all y on S^2 of min_i angle(y, p_i).
    This equals the maximum circumradius over all facets of the convex hull
    of the points (where each facet is a spherical triangle whose circumcenter
    is the farthest point from the origin in the spherical Voronoi cell).
    
    For a facet (triangle) with vertices p1, p2, p3 on the unit sphere,
    the circumcenter (in 3D) is the point equidistant (angularly) from p1, p2, p3.
    The angular distance from the circumcenter to any vertex is the circumradius.
    """
    # Compute convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use Delaunay if hull fails
        try:
            tri = Delaunay(points)
            facets = tri.simplices
        except Exception:
            return np.pi / 2  # fallback
    
    facets = hull.simplices
    
    max_circumradius = 0.0
    
    for facet in facets:
        p1, p2, p3 = points[facet[0]], points[facet[1]], points[facet[2]]
        
        # Compute circumcenter of the spherical triangle
        # The circumcenter is the intersection of the perpendicular bisectors
        # In 3D, for three unit vectors, the circumcenter direction is:
        # c = (p1 x p2) x (p2 x p3) ... no, let's use the formula
        
        # The circumcenter of a spherical triangle with vertices p1, p2, p3
        # can be found as the normalized cross product of the edge vectors' normals
        
        # Actually, the circumcenter is the point equidistant from p1, p2, p3.
        # It lies in the plane spanned by p1, p2, p3.
        # c = a*p1 + b*p2 + c*p3 where |c-p1|=|c-p2|=|c-p3| (angular distance)
        
        # For unit vectors, the circumcenter direction is:
        # c ∝ (p1 x p2) x (p2 x p3) ... let me think again
        
        # The circumcenter of three points on a sphere is the intersection of the sphere
        # with the plane equidistant from p1 and p2 (which is the plane through origin 
        # perpendicular to p1-p2), and similarly for other pairs.
        
        # Actually, the circumcenter is along the direction:
        # c = (p1 x p2) x (p2 x p3) ... no
        
        # Let me use the standard formula: for three unit vectors,
        # the circumcenter is proportional to:
        # c = (p1 x p2) x (p2 x p3) + cyclic terms... 
        
        # Simpler: the circumcenter is the normalized vector:
        # c = (p1 + p2 + p3) is NOT correct in general
        
        # Correct approach: the circumcenter is the solution to:
        # c · p1 = c · p2 = c · p3 (since angular distance is arccos(c·p_i))
        # and |c| = 1
        
        # So c is in the null space of:
        # [p1 - p2; p2 - p3] c = 0
        
        # c is proportional to (p1 - p2) x (p2 - p3)
        
        v1 = p1 - p2
        v2 = p2 - p3
        c_dir = np.cross(v1, v2)
        
        if np.linalg.norm(c_dir) < 1e-12:
            # Degenerate triangle
            continue
        
        c_dir = c_dir / np.linalg.norm(c_dir)
        
        # Check that this is the correct circumcenter (not the antipodal one)
        # The circumcenter should be "inside" the triangle, i.e., c·p_i > 0 for all i
        # (for small triangles). For larger triangles, we need to be careful.
        
        # Compute the circumradius
        cos_r = np.clip(c_dir @ p1, -1.0, 1.0)
        circumradius = np.arccos(cos_r)
        
        # Also check the other sign
        cos_r2 = np.clip(-c_dir @ p1, -1.0, 1.0)
        circumradius2 = np.arccos(cos_r2)
        
        # The covering radius uses the circumcenter that is farthest from the points
        # in the sense of being the center of the Voronoi cell
        # We want the larger circumradius (the one that represents the worst case)
        # Actually, for a facet of the convex hull, the circumcenter that matters
        # is the one on the same side as the facet's normal (outward)
        
        # The outward normal of the facet
        n = np.cross(p2 - p1, p3 - p1)
        n = n / np.linalg.norm(n)
        
        # The circumcenter should be in the direction of n (outward)
        if c_dir @ n < 0:
            c_dir = -c_dir
            circumradius = circumradius2
        else:
            circumradius = circumradius
        
        max_circumradius = max(max_circumradius, circumradius)
    
    return max_circumradius


def _covering_radius_grid(points, n_test=20000):
    """
    Compute covering radius using a fine grid of test points.
    This is an approximation but fast.
    """
    # Fibonacci spiral for test points
    i = np.arange(n_test) + 0.5
    z = 1.0 - 2.0 * i / n_test
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    test_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    cos_d = np.clip(test_points @ points.T, -1.0, 1.0)
    min_cos = np.max(cos_d, axis=1)
    min_angles = np.arccos(np.clip(min_cos, -1.0, 1.0))
    
    return np.max(min_angles)


def _objective(params):
    """
    Objective function: covering radius of the configuration.
    Uses the exact hull-based computation for precision.
    """
    thetas = params[0::2]
    phis = params[1::2]
    
    points = np.column_stack([
        np.sin(thetas) * np.cos(phis),
        np.sin(thetas) * np.sin(phis),
        np.cos(thetas)
    ])
    
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Use exact computation
    return _covering_radius_exact(points)


def _faster_objective(params):
    """
    Faster objective using grid approximation for initial optimization.
    """
    thetas = params[0::2]
    phis = params[1::2]
    
    points = np.column_stack([
        np.sin(thetas) * np.cos(phis),
        np.sin(thetas) * np.sin(phis),
        np.cos(thetas)
    ])
    
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return _covering_radius_grid(points, n_test=10000)


def _relax_repulsive(points, steps=100, step_size=0.02):
    """
    Repulsive force relaxation: each point is pushed away from others
    to maximize minimum distance, which helps with covering.
    """
    points = points.copy()
    
    for step in range(steps):
        forces = np.zeros_like(points)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = np.linalg.norm(points[i] - points[j])
                if d < 1e-6:
                    # Add small random perturbation
                    forces[i] += np.random.randn(3) * 0.01
                else:
                    # Repulsive force: stronger when close
                    force = (points[i] - points[j]) / (d * d)
                    forces[i] += force
        
        # Normalize forces and apply
        for i in range(N):
            f = forces[i]
            f_norm = np.linalg.norm(f)
            if f_norm > 0:
                f = f / f_norm
            # Project onto tangent plane
            tangent = f - np.dot(f, points[i]) * points[i]
            points[i] = points[i] + step_size * tangent
            points[i] = points[i] / np.linalg.norm(points[i])
    
    return points


def construct():
    # 1. Initial Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Repulsive Force Relaxation
    points_relaxed = _relax_repulsive(points0, steps=50, step_size=0.03)
    
    # 3. Convert to spherical coordinates
    thetas0 = np.arccos(np.clip(points_relaxed[:, 2], -1.0, 1.0))
    phis0 = np.arctan2(points_relaxed[:, 1], points_relaxed[:, 0])
    
    params0 = np.zeros(2 * N)
    params0[0::2] = thetas0
    params0[1::2] = phis0
    
    # 4. Coarse optimization with grid-based objective
    result1 = minimize(
        _faster_objective,
        params0,
        method='Nelder-Mead',
        options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False}
    )
    
    # 5. Fine optimization with exact objective
    result2 = minimize(
        _objective,
        result1.x,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
    )
    
    # Extract optimized points
    thetas = result2.x[0::2]
    phis = result2.x[1::2]
    
    points = np.column_stack([
        np.sin(thetas) * np.cos(phis),
        np.sin(thetas) * np.sin(phis),
        np.cos(thetas)
    ])
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # 6. Local refinement: perturb each point slightly and keep if it improves
    current_radius = _covering_radius_exact(points)
    
    for _ in range(100):
        improved = False
        for i in range(N):
            for trial in range(5):
                # Small random perturbation
                delta_theta = np.random.randn() * 0.005
                delta_phi = np.random.randn() * 0.005
                
                new_theta = thetas[i] + delta_theta
                new_phi = phis[i] + delta_phi
                
                # Keep within valid range
                new_theta = np.clip(new_theta, 1e-6, np.pi - 1e-6)
                
                old_theta = thetas[i]
                old_phi = phis[i]
                
                thetas[i] = new_theta
                phis[i] = new_phi
                
                test_points = np.column_stack([
                    np.sin(thetas) * np.cos(phis),
                    np.sin(thetas) * np.sin(phis),
                    np.cos(thetas)
                ])
                test_points = test_points / np.linalg.norm(test_points, axis=1, keepdims=True)
                
                new_radius = _covering_radius_exact(test_points)
                
                if new_radius < current_radius - 1e-10:
                    current_radius = new_radius
                    improved = True
                else:
                    thetas[i] = old_theta
                    phis[i] = old_phi
        
        if not improved:
            break
    
    # Final points
    points = np.column_stack([
        np.sin(thetas) * np.cos(phis),
        np.sin(thetas) * np.sin(phis),
        np.cos(thetas)
    ])
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points
# EVOLVE-BLOCK-END
