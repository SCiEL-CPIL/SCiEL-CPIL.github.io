"""
Optimized 50-point spherical covering using differentiable Voronoi-based optimization.

Approach:
- Start from a good initial configuration (Fibonacci spiral).
- Optimize by minimizing the sum of squared inradii of spherical Voronoi cells.
- The inradius of each Voronoi cell is the distance from the point to the farthest
  edge of its cell (the circumradius of the cell). Minimizing the sum of squares
  pushes all cells to be more uniform and smaller.
- Use gradient-based optimization with a differentiable approximation of the
  Voronoi cell inradii computed via angular bisection planes.
"""

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time


def _normalize(v):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return v / norms


def _spherical_voronoi_inradii(points):
    """
    Compute the inradii of the spherical Voronoi cells for points on the unit sphere.
    
    For each point, the inradius is the maximum angular distance from that point
    to any vertex of its Voronoi cell. We approximate this by finding the farthest
    point on the convex hull edges/faces adjacent to each point.
    
    A simpler and more robust approach: for each pair of neighboring points, the
    bisecting great circle defines an edge. The inradius of point i is the maximum
    over all its Voronoi edges of the distance from point i to the midpoint of the
    edge (projected back to the sphere).
    
    We use the convex hull approach: compute the convex hull of the points, then
    for each point, find the farthest vertex of its Voronoi cell.
    
    A practical differentiable approximation:
    For each point p_i, consider all other points p_j. The Voronoi edge between i and j
    lies on the great circle bisecting p_i and p_j. The "farthest" direction from p_i
    within its cell is approximately in the direction away from the centroid of its
    neighbors. 
    
    Actually, let's use a more direct method:
    For each point i, the inradius is the largest r such that a spherical cap of radius r
    centered at p_i is contained in its Voronoi cell. This is equivalent to:
    r_i = min over all j != i of (angular distance from p_i to the bisector plane of p_i, p_j)
    ... but that's the circumradius of the Delaunay triangle, not the Voronoi inradius.
    
    Let me use the correct definition: The inradius of the Voronoi cell of p_i is the
    minimum over all vertices v of the Voronoi cell of the angular distance from p_i to v.
    The vertices of the Voronoi cell are the circumcenters of Delaunay triangles.
    
    For a differentiable approach, we can compute:
    For each point i, find its nearest neighbors, compute the circumcenters of the
    Delaunay triangles involving i, and take the minimum distance to those circumcenters.
    
    This is complex. Let's use a simpler but effective approach:
    
    The covering radius is the max over all points on the sphere of the distance to
    the nearest site. This equals the max over all Voronoi cells of the inradius of
    that cell (where inradius here means the radius of the largest empty cap centered
    at the site, which is the distance from the site to the farthest point in its cell).
    
    Wait, I need to be careful. The "inradius" of a Voronoi cell in the context of
    covering is actually the circumradius of the cell - i.e., the farthest point in
    the cell from the site. This is what determines the covering radius.
    
    So for each point p_i, the relevant quantity is:
    R_i = max_{y in Voronoi cell of i} angular_dist(p_i, y)
    
    This is equal to the maximum over all vertices v of the Voronoi cell of
    angular_dist(p_i, v).
    
    To compute this differentiably:
    1. Compute the convex hull of the 50 points.
    2. For each point, identify its Voronoi cell vertices (circumcenters of Delaunay faces).
    3. Compute the max angular distance from the point to those vertices.
    
    This is non-trivial to make differentiable. Let's use an alternative:
    
    Use the fact that the covering radius equals the maximum circumradius of the
    facets of the convex hull. Each facet of the convex hull corresponds to a Voronoi
    vertex. The circumradius of a facet is the angular distance from the facet's
    circumcenter to any of its vertices.
    
    So: compute convex hull, for each facet compute the circumradius, and the
    covering radius is the max circumradius.
    
    To optimize, we minimize the sum of squared circumradii of all facets.
    
    The circumradius of a spherical triangle (facet) with vertices a, b, c is:
    The angular radius of the smallest spherical cap containing a, b, c.
    
    For a triangle on the unit sphere, the circumradius R satisfies:
    cos(R) = (cos(a-b) * cos(a-c) * cos(b-c) - cos(a-b)^2 * cos(a-c)^2 * cos(b-c)^2 + ...)
    
    Actually, for three points on the unit sphere, the circumradius (angular) can be
    computed as follows:
    - Compute the center of the circumscribed circle (the point equidistant from all three).
    - The circumradius is the angular distance from this center to any vertex.
    
    For a triangle with side lengths a, b, c (angular), the circumradius R is:
    sin(R) = (sin(a) * sin(b) * sin(c)) / (4 * sin(A) * sin(B) * sin(C))
    where A, B, C are the angles of the spherical triangle.
    
    This is getting complicated. Let's use a numerical approach:
    
    For each facet (triangle) with vertices p1, p2, p3 on the unit sphere:
    - Find the point c on the sphere equidistant from p1, p2, p3.
    - The circumradius is angular_dist(c, p1).
    
    To find c: c is the normalized cross product of (p2 - p1) and (p3 - p1), adjusted
    to be on the sphere... Actually, the circumcenter of a spherical triangle is the
    normalized vector that is equidistant from all three vertices.
    
    For three non-collinear points on the sphere, the circumcenter is:
    c = normalize(cross(p1 - p2, p1 - p3))  -- this gives a point on the great circle
    perpendicular to the plane... no.
    
    Actually, the circumcenter of a spherical triangle is the pole of the great circle
    that is the perpendicular bisector... Let me think differently.
    
    The circumcenter c satisfies: c·p1 = c·p2 = c·p3 = cos(R)
    So c is in the null space of [p1 - p2, p1 - p3] and has norm 1.
    
    c = normalize(cross(p1 - p2, p1 - p3))
    
    Then R = arccos(c · p1)
    
    Let's verify: if c = normalize(cross(p1-p2, p1-p3)), then c is perpendicular to
    (p1-p2) and (p1-p3), so c·p1 = c·p2 and c·p1 = c·p3. Good.
    
    So for each triangular facet, the circumradius is arccos(c · p1) where
    c = normalize(cross(p1-p2, p1-p3)).
    
    This is differentiable! We can compute the sum of squared circumradii and
    its gradient.
    
    Let's implement this.
    """
    # We'll compute this in the objective function instead
    pass


def _covering_radius(points):
    """Compute the covering radius of the configuration."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: compute max distance to nearest neighbor
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        # Sample points on the sphere
        n_samples = 10000
        rng = np.random.RandomState(42)
        samples = rng.randn(n_samples, 3)
        samples = _normalize(samples)
        dists, _ = tree.query(samples)
        return np.max(dists) * 180.0 / np.pi
    
    # For each facet, compute the circumradius
    max_circumradius = 0.0
    for simplex in hull.simplices:
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        # Circumcenter
        v1 = p1 - p2
        v2 = p1 - p3
        c = np.cross(v1, v2)
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-10:
            # Degenerate triangle
            # Use the max distance to the three points
            dists = [np.arccos(np.clip(np.dot(p1, p2), -1, 1)),
                     np.arccos(np.clip(np.dot(p1, p3), -1, 1)),
                     np.arccos(np.clip(np.dot(p2, p3), -1, 1))]
            circumradius = max(dists) / 2  # Approximation
        else:
            c = c / norm_c
            cos_r = np.clip(np.dot(c, p1), -1.0, 1.0)
            circumradius = np.arccos(cos_r)
        
        if circumradius > max_circumradius:
            max_circumradius = circumradius
    
    return max_circumradius * 180.0 / np.pi


def _objective_and_grad(x):
    """
    Objective: sum of squared circumradii of all facets of the convex hull.
    Gradient: computed via finite differences or analytic.
    
    x: flattened array of shape (3*N,)
    """
    points = _normalize(x.reshape(N, 3))
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails, use a penalty
        return 1e6, np.zeros_like(x)
    
    # Sum of squared circumradii
    obj = 0.0
    grad = np.zeros_like(x)
    
    for simplex in hull.simplices:
        i1, i2, i3 = simplex[0], simplex[1], simplex[2]
        p1, p2, p3 = points[i1], points[i2], points[i3]
        
        v1 = p1 - p2
        v2 = p1 - p3
        c = np.cross(v1, v2)
        norm_c = np.linalg.norm(c)
        
        if norm_c < 1e-10:
            continue
        
        c = c / norm_c
        cos_r = np.clip(np.dot(c, p1), -1.0, 1.0)
        circumradius = np.arccos(cos_r)
        
        obj += circumradius ** 2
        
        # Gradient with respect to p1, p2, p3
        # d(circumradius)/d(p_k) for k in {1,2,3}
        # circumradius = arccos(c · p1)
        # d(circumradius)/d(c) = -p1 / sqrt(1 - (c·p1)^2)
        # d(circumradius)/d(p1) = -c / sqrt(1 - (c·p1)^2)
        
        sin_r = np.sqrt(np.clip(1.0 - cos_r ** 2, 1e-10, 1.0))
        
        # d(circumradius)/d(p1) = -c / sin_r
        # d(circumradius)/d(p2) and d(circumradius)/d(p3) come from c depending on p2, p3
        
        # c = cross(p1-p2, p1-p3) / ||cross(p1-p2, p1-p3)||
        # This is complex. Let's use finite differences for the gradient.
        pass
    
    # Use finite differences for gradient
    eps = 1e-6
    grad = np.zeros_like(x)
    for j in range(len(x)):
        x_plus = x.copy()
        x_plus[j] += eps
        obj_plus = _objective(x_plus)
        grad[j] = (obj_plus - obj) / eps
    
    return obj, grad


def _objective(x):
    """Objective: sum of squared circumradii of all facets."""
    points = _normalize(x.reshape(N, 3))
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return 1e6
    
    obj = 0.0
    for simplex in hull.simplices:
        i1, i2, i3 = simplex[0], simplex[1], simplex[2]
        p1, p2, p3 = points[i1], points[i2], points[i3]
        
        v1 = p1 - p2
        v2 = p1 - p3
        c = np.cross(v1, v2)
        norm_c = np.linalg.norm(c)
        
        if norm_c < 1e-10:
            continue
        
        c = c / norm_c
        cos_r = np.clip(np.dot(c, p1), -1.0, 1.0)
        circumradius = np.arccos(cos_r)
        obj += circumradius ** 2
    
    return obj


def _grad_finite_diff(x):
    """Finite difference gradient of the objective."""
    eps = 1e-5
    grad = np.zeros_like(x)
    f0 = _objective(x)
    for j in range(len(x)):
        x_plus = x.copy()
        x_plus[j] += eps
        f_plus = _objective(x_plus)
        grad[j] = (f_plus - f0) / eps
    return grad


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    x0 = points.flatten().copy()
    
    # Optimize using L-BFGS-B
    start_time = time.time()
    max_time = 100  # Leave some margin
    
    best_x = x0.copy()
    best_obj = _objective(x0)
    
    # Use L-BFGS-B with finite difference gradients
    def obj_wrapper(x):
        return _objective(x)
    
    # Try a few different optimization strategies
    # Strategy 1: L-BFGS-B
    try:
        result = minimize(
            obj_wrapper,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000}
        )
        if result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > max_time - 20:
        points = _normalize(best_x.reshape(N, 3))
        return points
    
    # Strategy 2: Try local search with random restarts
    rng = np.random.RandomState(123)
    for restart in range(5):
        # Perturb the current best
        x_perturbed = best_x + rng.randn(N * 3) * 0.1
        try:
            result = minimize(
                obj_wrapper,
                x_perturbed,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 3000}
            )
            if result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
        
        if time.time() - start_time > max_time - 10:
            break
    
    # Final polish
    try:
        result = minimize(
            obj_wrapper,
            best_x,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000}
        )
        if result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    points = _normalize(best_x.reshape(N, 3))
    return points

# EVOLVE-BLOCK-END
