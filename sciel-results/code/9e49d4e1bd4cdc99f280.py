"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms

def _covering_radius(points):
    """
    Compute the covering radius by finding the largest circumradius of the
    facets of the convex hull. For a unit sphere, the covering radius is the
    maximum over all points y on the sphere of the angular distance to the
    nearest configuration point. This equals the maximum circumradius of the
    spherical Voronoi cells, which corresponds to the largest circumradius
    of the facets of the convex hull (in the dual sense).
    
    Actually, the covering radius is determined by the worst point on the sphere,
    which is the center of the largest empty cap. This corresponds to the facet
    of the convex hull with the largest circumradius (distance from origin to
    the plane of the facet, projected).
    
    More precisely: the covering radius is the maximum over all points on the sphere
    of the minimum angular distance to any of the N points. The worst point is
    the center of the largest empty spherical cap. This is equivalent to finding
    the facet of the convex hull of the N points that has the largest angular
    distance from the origin to its plane (i.e., the smallest |n| where n is the
    normal to the facet, since the distance from origin to the plane is 1/|n| for
    a unit sphere... actually let me think again).
    
    For points on the unit sphere, the convex hull facets are triangles (or other
    polygons) whose planes are at some distance d from the origin. The angular
    distance from the origin to the closest point on the facet's plane (within the
    sphere) is arccos(d). The covering radius is the maximum of arccos(d) over
    all facets where d < 1.
    
    Actually, the distance from origin to the plane of a facet with normal n (unit)
    is 1/|n| if the plane equation is n·x = c. For points on the unit sphere, the
    convex hull facets have planes that are at distance h from origin where h = 
    the minimum distance from origin to the plane. The angular radius of the cap
    centered at the facet's circumcenter is arccos(h).
    
    Let me use a more direct approach: for each facet (triangle) of the convex hull,
    compute the circumcenter of the triangle in 3D, normalize it to get the direction
    on the sphere, and compute the angular distance from that circumcenter to each
    of the three vertices. The maximum of these distances over all facets gives
    the covering radius.
    """
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use a grid-based estimate
        return _covering_radius_grid(points)
    
    max_radius = 0.0
    for simplex in hull.simplices:
        # Get the three points of this triangular facet
        tri = points[simplex]
        # Compute the circumcenter of the triangle in 3D
        a, b, c = tri
        # Circumcenter formula
        d = 2 * (a[0]*(b[1]*c[2] - c[1]*b[2]) - a[1]*(b[0]*c[2] - c[0]*b[2]) + a[2]*(b[0]*c[1] - c[0]*b[1]))
        if abs(d) < 1e-12:
            continue
        a2 = np.dot(a, a)
        b2 = np.dot(b, b)
        c2 = np.dot(c, c)
        ux = (a2*(b[1]*c[2] - c[1]*b[2]) - b2*(a[1]*c[2] - c[1]*a[2]) + c2*(a[1]*b[2] - b[1]*a[2])) / d
        uy = (a2*(c[0]*b[2] - b[0]*c[2]) - b2*(c[0]*a[2] - a[0]*c[2]) + c2*(b[0]*a[2] - a[0]*b[2])) / d
        uz = (a2*(b[0]*c[1] - c[0]*b[1]) - b2*(b[0]*a[1] - a[0]*b[1]) + c2*(c[0]*a[1] - a[0]*c[1])) / d
        circumcenter = np.array([ux, uy, uz])
        norm_cc = np.linalg.norm(circumcenter)
        if norm_cc < 1e-12:
            continue
        circumcenter = circumcenter / norm_cc
        # Compute angular distances from circumcenter to each vertex
        for v in tri:
            dot_val = np.dot(circumcenter, v)
            dot_val = np.clip(dot_val, -1.0, 1.0)
            angle = np.arccos(dot_val)
            if angle > max_radius:
                max_radius = angle
    return max_radius

def _covering_radius_grid(points, n_theta=360, n_phi=180):
    """Fallback grid-based covering radius estimate."""
    points = _normalize(points)
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    grid = np.column_stack([
        np.sin(THETA) * np.cos(PHI).flatten(),
        np.sin(THETA) * np.sin(PHI).flatten(),
        np.cos(THETA).flatten()
    ])
    # Compute distances
    dots = grid @ points.T  # shape (n_grid, N)
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=1)
    max_angle = np.arccos(min_dots).max()
    return max_angle

def _objective(params):
    """
    Objective: maximize the minimum distance from any point to its nearest neighbor.
    We minimize the negative of the minimum pairwise distance, or more directly,
    we minimize the covering radius.
    
    Actually, for optimization, it's better to use a smooth surrogate. Let's use
    the sum of exponentials of negative distances to encourage uniform distribution.
    """
    points = _normalize(params.reshape(N, 3))
    # Compute all pairwise dot products
    dots = points @ points.T
    np.fill_diagonal(dots, -1.0)  # Exclude self
    dots = np.clip(dots, -1.0, 1.0)
    # Min distance to nearest neighbor (excluding self)
    min_dots = np.min(dots, axis=1)
    # We want to maximize the minimum distance, so minimize -min_dots
    # But this is nonsmooth. Use a soft version:
    # Sum of exp(-k * (1 - min_dot)) for each point
    k = 5.0
    vals = 1.0 - min_dots  # This is related to angular distance squared approximately
    return np.sum(np.exp(k * vals))

def _gradient(params):
    """
    Compute gradient of the objective function.
    """
    points = _normalize(params.reshape(N, 3))
    dots = points @ points.T
    np.fill_diagonal(dots, -1.0)
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.min(dots, axis=1)
    k = 5.0
    vals = 1.0 - min_dots
    exp_vals = np.exp(k * vals)
    # Gradient of exp(k*(1-min_dot)) w.r.t. min_dot is -k*exp(k*(1-min_dot))
    # min_dot for point i is the max dot product with other points
    # This is complex to differentiate exactly, so we'll use numerical gradient or
    # a simpler approach.
    
    # Alternative: use the gradient of the sum of pairwise repulsive forces
    # Force on point i from point j is proportional to (1 - dot_ij) * (point_i - dot_ij * point_j)
    # This pushes points apart.
    
    grad = np.zeros_like(points)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dot_ij = np.dot(points[i], points[j])
            dot_ij = np.clip(dot_ij, -1.0, 1.0)
            # Repulsive force
            force = (1.0 - dot_ij) * (points[i] - dot_ij * points[j])
            grad[i] += force
    # Project gradient onto tangent space
    for i in range(N):
        grad[i] -= np.dot(grad[i], points[i]) * points[i]
    return grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Phase 1: Lloyd's algorithm (spherical k-means)
    # This iteratively assigns points to nearest centroid and moves centroids
    # to the center of their Voronoi cells.
    for iteration in range(20):
        if time.time() > deadline - 30.0:
            break
        
        # Assign each point on a fine grid to nearest configuration point
        # Actually, for Lloyd's on the sphere, we need to compute the spherical
        # Voronoi cells. A simpler approach: use the convex hull facets.
        
        # Alternative: use a repulsion-based relaxation
        # Move each point in the direction of the average of its neighbors'
        # repulsive forces.
        
        # Compute repulsive forces
        dots = points @ points.T
        np.fill_diagonal(dots, -1.0)
        dots = np.clip(dots, -1.0, 1.0)
        
        # Force on each point
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dot_ij = dots[i, j]
                # Repulsive force proportional to (1 - dot_ij)
                force = (1.0 - dot_ij) * (points[i] - dot_ij * points[j])
                forces[i] += force
        
        # Project forces onto tangent space
        for i in range(N):
            forces[i] -= np.dot(forces[i], points[i]) * points[i]
        
        # Move points
        step_size = 0.01
        points = points + step_size * forces
        points = _normalize(points)
        
        # Evaluate
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
    
    # Phase 2: L-BFGS optimization
    if time.time() < deadline - 20.0:
        # Use the best points as initial guess
        x0 = best_points.flatten()
        
        # Use a smooth objective: minimize the sum of exp(k * (1 - min_dot))
        # This encourages all points to have large minimum distances.
        
        def obj_and_grad(x):
            pts = _normalize(x.reshape(N, 3))
            dots = pts @ pts.T
            np.fill_diagonal(dots, -1.0)
            dots = np.clip(dots, -1.0, 1.0)
            min_dots = np.min(dots, axis=1)
            k = 3.0
            vals = 1.0 - min_dots
            obj = np.sum(np.exp(k * vals))
            
            # Gradient
            exp_vals = np.exp(k * vals)
            # For each point i, the gradient w.r.t. x_i
            grad = np.zeros_like(pts)
            for i in range(N):
                # Find which j gives the min dot for point i
                j_min = np.argmin(dots[i, :])
                # Gradient of exp(k*(1 - dot_ij)) w.r.t. x_i
                # d/dx_i exp(k*(1 - dot(x_i, x_j))) = -k * exp(k*(1-dot)) * x_j
                # But we also need to account for the constraint that x_i is on the sphere
                grad_i = -k * exp_vals[i] * pts[j_min]
                # Project onto tangent space
                grad_i -= np.dot(grad_i, pts[i]) * pts[i]
                grad[i] = grad_i
            return obj, grad.flatten()
        
        try:
            result = minimize(
                obj_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success and time.time() < deadline - 10.0:
                new_points = _normalize(result.x.reshape(N, 3))
                new_radius = _covering_radius(new_points)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_points = new_points
        except Exception:
            pass
    
    # Phase 3: Final polish with projected gradient descent on covering radius
    if time.time() < deadline - 5.0:
        points = best_points.copy()
        for iteration in range(50):
            if time.time() > deadline - 3.0:
                break
            
            # Compute covering radius and find worst facet
            try:
                hull = ConvexHull(points)
            except Exception:
                break
            
            max_radius = 0.0
            worst_facet = None
            worst_circumcenter = None
            
            for simplex in hull.simplices:
                tri = points[simplex]
                a, b, c = tri
                d = 2 * (a[0]*(b[1]*c[2] - c[1]*b[2]) - a[1]*(b[0]*c[2] - c[0]*b[2]) + a[2]*(b[0]*c[1] - c[0]*b[1]))
                if abs(d) < 1e-12:
                    continue
                a2 = np.dot(a, a)
                b2 = np.dot(b, b)
                c2 = np.dot(c, c)
                ux = (a2*(b[1]*c[2] - c[1]*b[2]) - b2*(a[1]*c[2] - c[1]*a[2]) + c2*(a[1]*b[2] - b[1]*a[2])) / d
                uy = (a2*(c[0]*b[2] - b[0]*c[2]) - b2*(c[0]*a[2] - a[0]*c[2]) + c2*(b[0]*a[2] - a[0]*b[2])) / d
                uz = (a2*(b[0]*c[1] - c[0]*b[1]) - b2*(b[0]*a[1] - a[0]*b[1]) + c2*(c[0]*a[1] - a[0]*c[1])) / d
                circumcenter = np.array([ux, uy, uz])
                norm_cc = np.linalg.norm(circumcenter)
                if norm_cc < 1e-12:
                    continue
                circumcenter = circumcenter / norm_cc
                for v in tri:
                    dot_val = np.dot(circumcenter, v)
                    dot_val = np.clip(dot_val, -1.0, 1.0)
                    angle = np.arccos(dot_val)
                    if angle > max_radius:
                        max_radius = angle
                        worst_facet = simplex
                        worst_circumcenter = circumcenter
            
            if worst_facet is None:
                break
            
            # Move the vertices of the worst facet toward the circumcenter
            # to reduce the radius
            alpha = 0.005
            for idx in worst_facet:
                points[idx] = points[idx] + alpha * (worst_circumcenter - points[idx])
            points = _normalize(points)
            
            # Evaluate
            radius = _covering_radius(points)
            if radius < best_radius:
                best_radius = radius
                best_points = points.copy()
    
    return best_points
# EVOLVE-BLOCK-END
