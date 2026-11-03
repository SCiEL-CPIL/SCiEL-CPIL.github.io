"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_SEED = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time


def _normalize(points):
    """Normalize rows to unit length, handling zero vectors."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms


def _covering_radius(pts):
    """
    Compute the covering radius (in radians) of a set of unit vectors.
    Uses the convex hull: the covering radius is the max circumradius of
    each facet. For a facet with outward normal n and vertices, the
    circumradius (angular distance from sphere center to farthest point
    on that facet's cap) is pi - min_angle where min_angle is the
    minimum angular distance from the facet's circumcenter to the
    nearest configuration point.
    
    Actually, the covering radius is the maximum over all facets of the
    angular radius of the spherical cap that has the facet as its boundary.
    For a planar facet with outward normal n and inradius r (distance from
    center of sphere to the plane), the angular radius is arccos(r) if
    the facet's circumcircle is fully on the sphere. But more precisely:
    
    The covering radius is the largest t such that there exists a point
    on the sphere at angular distance >= t from all chosen points.
    This equals the max over facets of the circumradius of the spherical
    polygon corresponding to that facet.
    
    For a convex polytope inscribed in the unit sphere, the covering radius
    is the maximum, over all faces, of the angular distance from the
    face's circumcenter (on the sphere) to the farthest vertex of that face.
    
    A simpler approach: the covering radius is the maximum over all points
    y on the sphere of min_i angle(y, p_i). This is achieved at a vertex
    of the Voronoi diagram on the sphere, which corresponds to a circumcenter
    of some subset of points. For the convex hull approach:
    
    The covering radius = max over facets F of (pi - min_{p in F} angle(n_F, p))
    where n_F is the outward unit normal of the facet... no, that's not right either.
    
    Let me think more carefully. The farthest point from all configuration
    points on the sphere lies at the circumcenter of some face of the
    convex hull. For a face with vertices v_1, ..., v_k (unit vectors),
    the circumcenter c is the unit vector equidistant (in angle) from all
    v_i. The angular distance from c to each v_i is the circumradius of
    that face. The covering radius is the max over all faces of this
    circumradius.
    
    To compute the circumcenter of a face: for a triangle (v1, v2, v3),
    the circumcenter is proportional to v1 + v2 + v3 adjusted... Actually
    for points on the unit sphere, the circumcenter of a spherical triangle
    is not simply the normalized sum.
    
    Alternative: the covering radius can be computed as the maximum, over
    all triples (or quadruples) of points, of the circumradius. But a more
    efficient method:
    
    For each facet of the convex hull, compute the circumradius. The
    circumradius of a planar polygon inscribed in a circle of radius R
    (where R is the distance from the sphere center to the circumcircle)
    at height h from the center... 
    
    Actually, let's use a different approach. The covering radius is the
    maximum t such that a spherical cap of angular radius t can be placed
    anywhere without containing any of the N points. Equivalently, it's
    the minimum over all unit vectors y of the maximum over i of the
    angular distance from y to p_i... no wait, it's the maximum over y of
    the minimum over i of the angular distance.
    
    For the convex hull: the farthest point y from all p_i lies at the
    circumcenter of some face of conv(p_1, ..., p_N). For a face with
    vertices v_1,...,v_k on the unit sphere, the circumcenter c is the
    unit vector that maximizes the minimum angular distance to the v_i
    within that face's spherical polygon.
    
    A practical computation: for each face, find the point on the sphere
    that is equidistant from all vertices of the face (the spherical
    circumcenter). For a triangle, this is the intersection of the
    perpendicular bisectors. In 3D, the circumcenter of a triangle
    (v1, v2, v3) on the unit sphere is:
    
    c ∝ v1 * |v2-v3|^2 + v2 * |v3-v1|^2 + v3 * |v1-v2|^2
    ... no, that's the planar circumcenter formula applied in 3D.
    
    Actually, for three points on the unit sphere, the circumcenter is
    the normalized vector:
    c = normalize(v1 + v2 + v3) only if they form a right triangle on
    the sphere, which is not general.
    
    The correct formula: the circumcenter of a triangle with vertices
    v1, v2, v3 (all unit vectors) is the unit vector c such that
    c · v1 = c · v2 = c · v3. This means c is in the direction of:
    c ∝ (v2 - v3) × (v1 - v3) ... no.
    
    c · v1 = c · v2 => c · (v1 - v2) = 0
    c · v1 = c · v3 => c · (v1 - v3) = 0
    So c is perpendicular to (v1-v2) and (v1-v3), hence:
    c ∝ (v1-v2) × (v1-v3)
    
    Wait, that gives the normal to the plane containing v1, v2, v3, not
    the circumcenter. The circumcenter must also satisfy c · v1 = c · v2 = c · v3.
    
    Let me set up: c = a*v1 + b*v2 + c*v3 (linear combination).
    c · v1 = c · v2: a + b*(v1·v2) + c*(v3·v1) = b + a*(v1·v2) + c*(v3·v2)
    This gets complicated. 
    
    Simpler: use the fact that for the covering radius, we can use the
    convex hull and for each face, compute the circumradius as follows:
    
    For a face with vertices v_1, ..., v_k on the unit sphere, the
    circumradius R satisfies sin(R) = r / h where r is the circumradius
    of the planar polygon and h is the distance from the origin to the
    plane of the face. Actually, R = arccos(h / (h^2 + r^2)^{1/2})... 
    
    Let me just use a direct numerical approach for each face:
    The circumcenter c of a face is the unit vector that maximizes
    min_i angular_dist(c, v_i) over the face. For a triangle, it's the
    unique point equidistant from all three vertices.
    
    For a triangle v1, v2, v3:
    c ∝ (v1 - v2) × (v1 - v3) gives the plane normal, not the circumcenter.
    
    The circumcenter of a spherical triangle can be found as the pole of
    the great circle through the midpoints... 
    
    OK, let me use a completely different, robust approach:
    The covering radius is the maximum over all unit vectors y of
    min_i arccos(y · p_i). This is equivalent to: find the maximum t
    such that there exists y with y · p_i <= cos(t) for all i.
    
    For the convex hull approach, the maximum is achieved at a vertex of
    the arrangement of great circles, i.e., at a circumcenter of some
    subset of points. For N=50, checking all subsets is infeasible.
    
    Practical approach: compute the convex hull, and for each face,
    compute the circumradius numerically.
    
    For a face with vertices v_1, ..., v_k (unit vectors), the circumcenter
    c is the unit vector equidistant from all v_i. We can find it by:
    1. Compute the plane of the face: normal n = normalize(sum of cross products)
    2. The circumcenter lies in the plane containing the origin and
       perpendicular to the face plane... no.
    
    Actually, the circumcenter c satisfies c · v_i = R for all i (where R = cos(angular_radius)).
    So c is in the intersection of the hyperplanes c · v_i = R.
    
    For a triangle: c · v1 = c · v2 = c · v3 = R.
    c is in the span of {v1, v2, v3} (since the constraints define a line
    in 3D intersected with the unit sphere).
    
    c = a*v1 + b*v2 + c*v3
    c·v1 = a + b*(v1·v2) + c*(v3·v1) = R
    c·v2 = a*(v1·v2) + b + c*(v3·v2) = R
    c·v3 = a*(v3·v1) + b*(v3·v2) + c = R
    
    This is a 3x3 linear system for a, b, c given R, but R is unknown.
    However, c·v1 = c·v2 = c·v3 implies:
    c·(v1-v2) = 0 and c·(v1-v3) = 0
    So c is parallel to (v1-v2) × (v1-v3).
    
    Let n = (v1-v2) × (v1-v3), normalized.
    Then c = ±n / ||n|| (unit vector).
    The angular radius is arccos(|c · v1|) = arccos(|n · v1| / ||n||).
    
    Wait, that's the pole of the great circle through v1, v2, v3, which
    IS the circumcenter of the spherical triangle! Because the pole is
    equidistant from all points on the great circle, and v1, v2, v3 are
    on that great circle.
    
    So for a triangle face, the circumcenter is the normalized normal to
    the plane of the triangle, and the circumradius is arccos(|n·v1|/||n||).
    
    For a quadrilateral or higher face, the circumcenter is more complex,
    but we can triangulate the face and take the max circumradius over
    the triangles.
    
    Let me implement this.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: brute force on a grid
        # This should not happen for well-distributed points
        return np.pi / 2  # conservative upper bound
    
    max_rad = 0.0
    for simplex in hull.simplices:
        # simplex is a set of 3 indices for a triangular face
        v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        # Normal to the plane of the triangle
        n = np.cross(v1 - v2, v1 - v3)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            continue
        n = n / norm_n
        # The circumcenter is ±n; the angular radius is arccos(|n·v1|)
        dot_val = abs(n @ v1)
        dot_val = min(dot_val, 1.0)  # clamp for numerical safety
        rad = np.arccos(dot_val)
        if rad > max_rad:
            max_rad = rad
    
    return max_rad


def _covering_radius_degrees(pts):
    """Covering radius in degrees."""
    return np.degrees(_covering_radius(pts))


def _objective_and_grad(x, pts_flat):
    """
    Objective: soft-max of angular distances from a test point to all pts.
    We want to minimize the covering radius, so we use a smooth surrogate.
    """
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    # Use the convex hull based covering radius as the objective
    # But this is non-smooth, so use a soft version
    # For now, use the actual covering radius
    return _covering_radius(pts)


def _soft_covering_radius(pts, beta=50.0):
    """
    Smooth approximation of the covering radius using log-sum-exp.
    For each point y on a fine grid, compute min_i dist(y, p_i),
    then take a soft-max over the grid.
    """
    pts = _normalize(pts)
    # Create a grid of test points on the sphere
    # Use a moderate grid for speed
    n_theta = 20
    n_phi = 40
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    Y = np.column_stack([
        np.sin(T) * np.cos(P).ravel(),
        np.sin(T) * np.sin(P).ravel(),
        np.cos(T).ravel()
    ])
    Y = _normalize(Y)
    # Compute distances: M[i,j] = angular dist from Y[i] to pts[j]
    dots = Y @ pts.T  # (n_y, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # (n_y, N)
    # For each y, find min distance to any p_i
    min_dists = np.min(dists, axis=1)  # (n_y,)
    # Soft-max: approximate max with log-sum-exp
    beta = 50.0
    max_d = np.max(min_dists)
    soft_max = max_d + np.log(np.sum(np.exp(beta * (min_dists - max_d)))) / beta
    return soft_max


def _soft_covering_grad(x, beta=50.0):
    """Gradient of the soft covering radius w.r.t. x."""
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    n_theta = 20
    n_phi = 40
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    Y = np.column_stack([
        np.sin(T) * np.cos(P).ravel(),
        np.sin(T) * np.sin(P).ravel(),
        np.cos(T).ravel()
    ])
    Y = _normalize(Y)
    dots = Y @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    # Which point is nearest for each y
    nearest_idx = np.argmin(dists, axis=1)
    
    max_d = np.max(min_dists)
    weights = np.exp(beta * (min_dists - max_d))
    weights /= np.sum(weights)
    
    # Gradient: d/dx of sum_i w_i * dist(y_i, p_{nearest_i})
    # dist = arccos(y · p), d/dp arccos(y·p) = -y / sqrt(1 - (y·p)^2)
    grad = np.zeros_like(x)
    for i in range(len(Y)):
        y = Y[i]
        j = nearest_idx[i]
        d = dots[i, j]
        d = np.clip(d, -0.999999, 0.999999)
        denom = np.sqrt(1.0 - d * d)
        if denom < 1e-12:
            continue
        grad[j * 3:(j + 1) * 3] += weights[i] * (-y / denom)
    
    # Project gradient to tangent space (since we normalize)
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    p_unit = pts / norms
    # Remove radial component
    radial = (grad.reshape(N, 3) * p_unit).sum(axis=1, keepdims=True)
    grad_tangent = grad.reshape(N, 3) - radial * p_unit
    return grad_tangent.ravel()


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial point: Fibonacci spiral
    def fib_spiral():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = fib_spiral()
    best_radius = _covering_radius(best_pts)
    
    # Multi-start local optimization
    np.random.seed(_SEED)
    
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        if restart == 0:
            x0 = best_pts.copy()
        else:
            # Random start with some structure
            x0 = np.random.randn(N, 3)
            x0 = _normalize(x0)
        
        # Use Nelder-Mead on the soft covering radius
        def obj(x):
            pts = _normalize(x.reshape(N, 3))
            return _soft_covering_radius(pts, beta=30.0)
        
        try:
            res = minimize(
                obj,
                x0.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
        except Exception:
            continue
        
        x_opt = res.x.reshape(N, 3)
        x_opt = _normalize(x_opt)
        
        # Compute actual covering radius
        rad = _covering_radius(x_opt)
        if rad < best_radius:
            best_radius = rad
            best_pts = x_opt.copy()
    
    # Final polish: use the convex hull covering radius directly with
    # a few more Nelder-Mead iterations
    if time.time() < deadline - 5.0:
        def obj_exact(x):
            pts = _normalize(x.reshape(N, 3))
            return _covering_radius(pts)
        
        try:
            res = minimize(
                obj_exact,
                best_pts.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            x_opt = res.x.reshape(N, 3)
            x_opt = _normalize(x_opt)
            rad = _covering_radius(x_opt)
            if rad < best_radius:
                best_radius = rad
                best_pts = x_opt.copy()
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
