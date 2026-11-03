"""Rewritten program: Riesz s-energy (s=12) minimisation on the sphere via L-BFGS-B."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
RIESZ_S = 12.0
WALL_CLOCK_LIMIT = 140.0  # safety margin below 150s


def _riesz_energy_and_grad(points, s=RIESZ_S):
    """
    Compute the Riesz s-energy and its gradient for N points on the unit sphere.
    Energy: sum_{i != j} |p_i - p_j|^{-s}
    Gradient w.r.t. p_i: -s * sum_{j != i} |p_i - p_j|^{-s-2} * (p_i - p_j)
    """
    n = points.shape[0]
    # Pairwise differences: diff[i,j] = p_i - p_j
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Avoid division by zero on the diagonal
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)

    # Energy
    mask = ~np.eye(n, dtype=bool)
    energy = np.sum(dist[mask] ** (-s))

    # Gradient: grad_i = -s * sum_{j != i} dist_{ij}^{-(s+2)} * (p_i - p_j)
    # dist^{-(s+2)} = dist^{-s} / dist^2
    inv_dist_s = dist ** (-s)
    np.fill_diagonal(inv_dist_s, 0.0)
    inv_dist_s2 = inv_dist_s / dist_sq
    np.fill_diagonal(inv_dist_s2, 0.0)

    # grad[i] = -s * sum_j inv_dist_s2[i,j] * diff[i,j]
    grad = -s * np.sum(inv_dist_s2[:, :, np.newaxis] * diff, axis=1)

    return energy, grad


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _compute_covering_radius(points):
    """
    Compute the covering radius in degrees.
    Uses the convex hull facets approach: for each facet (triangle) of the convex hull,
    find the circumcenter and check if it's on the unit sphere (empty cap).
    The covering radius is the max angular distance from any sphere point to nearest config point.
    
    For a set of points on the unit sphere, the covering radius equals the maximum over all
    empty spherical caps (defined by 3 mutually nearest points forming a Delaunay triangle)
    of the circumradius of that triangle.
    
    A simpler and robust approach: use scipy.spatial.Delaunay on the 3D points, then for each
    tetrahedron, check the opposite face. But for points on a sphere, the convex hull facets
    are triangles (or higher, but we triangulate). The covering radius is the max over all
    hull facets of the distance from the facet's circumcenter (projected to sphere) to the
    nearest point, which equals the circumradius of the spherical triangle.
    
    Actually, the most robust method: for each point y on the sphere, find the nearest config point.
    The max of this is the covering radius. We can approximate by checking:
    1. The config points themselves (distance 0)
    2. The circumcenters of all hull facets
    
    For points on the unit sphere, the farthest point from the set is either a config point (dist 0)
    or the circumcenter of some empty spherical triangle. So we compute the circumradius for each
    hull facet.
    """
    from scipy.spatial import ConvexHull
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: brute-force on a fine grid
        theta = np.linspace(0, np.pi, 200)
        phi = np.linspace(0, 2 * np.pi, 400)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        Y = np.column_stack([
            np.sin(TH) * np.cos(PH),
            np.sin(TH) * np.sin(PH),
            np.cos(TH)
        ]).reshape(-1, 3)
        # Normalize
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        # For each Y, find min distance to points
        # dist_sq = |Y|^2 + |P|^2 - 2*Y.P = 2 - 2*Y.P
        YP = Y @ points.T
        min_dist_sq = np.min(2.0 - 2.0 * YP, axis=1)
        max_dist = np.sqrt(np.max(min_dist_sq))
        return np.degrees(np.arccos(np.clip(1.0 - 0.5 * max_dist * max_dist, -1.0, 1.0)))
    
    # For each hull facet, compute the circumradius
    facets = hull.simplices
    max_ang = 0.0
    for tri in facets:
        a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
        # Circumcenter of triangle on the unit sphere
        # The circumcenter of a spherical triangle is the point equidistant from a, b, c.
        # It lies in the plane spanned by a, b, c. The circumcenter direction is:
        # n = (a x b) x (b x c) / |...| ... actually let's use the formula for circumcenter of 3 points on sphere.
        # The circumcenter O satisfies O.a = O.b = O.c and |O|=1.
        # O is in the span of a, b, c. Let O = alpha*a + beta*b + gamma*c.
        # O.a = O.b = O.c => alpha + beta*(a.b) + gamma*(a.c) = beta*(b.a) + 1 + gamma*(b.c) = ...
        # This is a 3x3 linear system. Let's solve it.
        M = np.array([
            [a @ a, a @ b, a @ c],
            [b @ a, b @ b, b @ c],
            [c @ a, c @ b, c @ c]
        ])
        rhs = np.array([a @ a, b @ b, c @ c])
        try:
            coeffs = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            continue
        O = coeffs[0] * a + coeffs[1] * b + coeffs[2] * c
        O_norm = np.linalg.norm(O)
        if O_norm < 1e-15:
            continue
        O = O / O_norm
        # Check that O is on the unit sphere and the cap is empty (O is on the same side as the hull)
        # The circumradius is the angle between O and any of a, b, c
        cos_ang = np.clip(a @ O, -1.0, 1.0)
        ang = np.arccos(cos_ang)
        if ang > max_ang:
            max_ang = ang
    return np.degrees(max_ang)


def _run_optimization(initial_points, deadline):
    """
    Run L-BFGS-B optimization of the Riesz energy, projecting back to the sphere after each step.
    Returns the optimized points.
    """
    x0 = initial_points.flatten()
    
    def objective(x):
        points = x.reshape(N, 3)
        points = _project_to_sphere(points)
        energy, grad = _riesz_energy_and_grad(points)
        # Gradient on the sphere: project gradient to tangent space
        # grad_tangent = grad - (grad . p) * p
        grad_proj = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        return energy, grad_proj.flatten()
    
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
    )
    
    x_opt = result.x.reshape(N, 3)
    x_opt = _project_to_sphere(x_opt)
    return x_opt


def construct():
    start_time = time.time()
    deadline = start_time + WALL_CLOCK_LIMIT
    
    # Generate diverse initial configurations
    rng = np.random.default_rng(42)
    
    configs = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    configs.append(fib)
    
    # 2. Random
    for _ in range(6):
        pts = rng.normal(size=(N, 3))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts)
    
    best_points = None
    best_radius = np.inf
    
    for idx, init in enumerate(configs):
        if time.time() > deadline - 20.0:
            break
        
        try:
            optimized = _run_optimization(init, deadline)
            # Compute covering radius
            radius = _compute_covering_radius(optimized)
            if radius < best_radius:
                best_radius = radius
                best_points = optimized.copy()
        except Exception:
            continue
    
    if best_points is None:
        # Fallback to Fibonacci
        return fib
    
    # Final polish: run one more optimization from the best point
    if time.time() < deadline - 10.0:
        try:
            polished = _run_optimization(best_points, deadline)
            radius_polished = _compute_covering_radius(polished)
            if radius_polished < best_radius:
                best_points = polished
        except Exception:
            pass
    
    return best_points
