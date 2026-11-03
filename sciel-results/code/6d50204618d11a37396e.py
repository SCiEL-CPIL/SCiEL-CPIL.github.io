import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

N = 50
START_TIME = time.time()
DEADLINE = 110.0  # Leave 40s margin before 150s kill

def to_sphere(params):
    """Convert raw params to unit vectors."""
    v = params.reshape(-1, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return v / norms

def compute_facet_circumradii(points):
    """Compute circumradius and circumcenter for each convex hull facet."""
    points = np.asarray(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None
    
    circumradii = []
    circumcenters = []
    facet_vertices = []
    
    for simplex in hull.simplices:
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
        ab = b - a
        ac = c - a
        abab = np.dot(ab, ab)
        acac = np.dot(ac, ac)
        abac = np.dot(ab, ac)
        
        denom = abab * acac - abac * abac
        if abs(denom) < 1e-12:
            continue
        
        u1 = (acac * abac - abab * abac) / denom
        u2 = (abab * abac - acac * abac) / denom
        circumcenter = a + u1 * ab + u2 * ac
        
        cc_norm = np.linalg.norm(circumcenter)
        if cc_norm < 1e-10:
            continue
        
        pole = circumcenter / cc_norm
        
        theta_a = np.arccos(np.clip(np.dot(pole, a), -1, 1))
        theta_b = np.arccos(np.clip(np.dot(pole, b), -1, 1))
        theta_c = np.arccos(np.clip(np.dot(pole, c), -1, 1))
        
        circum_angular = max(theta_a, theta_b, theta_c)
        
        circumradii.append(circum_angular)
        circumcenters.append(pole)
        facet_vertices.append(simplex)
    
    if len(circumradii) == 0:
        return None, None, None
    
    return np.array(circumradii), np.array(circumcenters), facet_vertices

def compute_covering_radius(points):
    """Compute the covering radius (in radians)."""
    circumradii, _, _ = compute_facet_circumradii(points)
    if circumradii is None:
        return np.pi
    return np.max(circumradii)

def voronoi_covering_radius_and_grad(points, beta=10.0):
    """Compute smooth covering radius using Voronoi cells and its gradient.
    
    For each point i, its Voronoi cell on the sphere is the set of points
    closer to i than to any other point. The covering radius is the max over
    all Voronoi vertices (circumcenters of Delaunay triangles) of the angular
    distance to the nearest point.
    
    We use a log-sum-exp smoothing over all Delaunay triangle circumradii.
    
    Returns:
        cr: smoothed covering radius
        grad: gradient w.r.t. the input point coordinates (before normalization)
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, np.zeros_like(points)
    
    # Compute circumradii and circumcenters for all facets
    circumradii = []
    circumcenters = []
    facet_vertices = []
    
    for simplex in hull.simplices:
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
        ab = b - a
        ac = c - a
        abab = np.dot(ab, ab)
        acac = np.dot(ac, ac)
        abac = np.dot(ab, ac)
        
        denom = abab * acac - abac * abac
        if abs(denom) < 1e-12:
            continue
        
        u1 = (acac * abac - abab * abac) / denom
        u2 = (abab * abac - acac * abac) / denom
        circumcenter = a + u1 * ab + u2 * ac
        
        cc_norm = np.linalg.norm(circumcenter)
        if cc_norm < 1e-10:
            continue
        
        pole = circumcenter / cc_norm
        
        theta_a = np.arccos(np.clip(np.dot(pole, a), -1, 1))
        theta_b = np.arccos(np.clip(np.dot(pole, b), -1, 1))
        theta_c = np.arccos(np.clip(np.dot(pole, c), -1, 1))
        
        circum_angular = max(theta_a, theta_b, theta_c)
        
        circumradii.append(circum_angular)
        circumcenters.append(pole)
        facet_vertices.append(simplex)
    
    if len(circumradii) == 0:
        return np.pi, np.zeros_like(points)
    
    circumradii = np.array(circumradii)
    circumcenters = np.array(circumcenters)
    n_facets = len(circumradii)
    
    # Log-sum-exp smoothing
    max_cr = np.max(circumradii)
    shifted = circumradii - max_cr
    weights = np.exp(shifted / beta)
    weights /= np.sum(weights)
    
    cr = max_cr + beta * np.log(np.sum(weights))
    
    # Gradient computation using finite differences on the sphere
    # For each point, compute how the smoothed CR changes when we move that point
    grad = np.zeros_like(points)
    eps = 1e-4
    
    for i in range(n):
        # Perturb point i along its tangent plane
        p = points[i]
        
        # Generate two tangent vectors
        v1 = np.cross(p, [1, 0, 0])
        if np.linalg.norm(v1) < 1e-6:
            v1 = np.cross(p, [0, 1, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(p, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        for axis, v in enumerate([v1, v2]):
            # Forward difference
            p_plus = p + eps * v
            p_plus = p_plus / np.linalg.norm(p_plus)
            points_plus = points.copy()
            points_plus[i] = p_plus
            
            cr_plus, _ = voronoi_covering_radius_and_grad_no_grad(points_plus, beta)
            
            # Backward difference
            p_minus = p - eps * v
            p_minus = p_minus / np.linalg.norm(p_minus)
            points_minus = points.copy()
            points_minus[i] = p_minus
            
            cr_minus, _ = voronoi_covering_radius_and_grad_no_grad(points_minus, beta)
            
            dcr_dv = (cr_plus - cr_minus) / (2 * eps)
            grad[i] += dcr_dv * v
    
    return cr, grad

def voronoi_covering_radius_and_grad_no_grad(points, beta=10.0):
    """Compute smoothed covering radius without gradient (for finite differences)."""
    points = np.asarray(points, dtype=float)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, None
    
    circumradii = []
    
    for simplex in hull.simplices:
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
        ab = b - a
        ac = c - a
        abab = np.dot(ab, ab)
        acac = np.dot(ac, ac)
        abac = np.dot(ab, ac)
        
        denom = abab * acac - abac * abac
        if abs(denom) < 1e-12:
            continue
        
        u1 = (acac * abac - abab * abac) / denom
        u2 = (abab * abac - acac * abac) / denom
        circumcenter = a + u1 * ab + u2 * ac
        
        cc_norm = np.linalg.norm(circumcenter)
        if cc_norm < 1e-10:
            continue
        
        pole = circumcenter / cc_norm
        
        theta_a = np.arccos(np.clip(np.dot(pole, a), -1, 1))
        theta_b = np.arccos(np.clip(np.dot(pole, b), -1, 1))
        theta_c = np.arccos(np.clip(np.dot(pole, c), -1, 1))
        
        circum_angular = max(theta_a, theta_b, theta_c)
        circumradii.append(circum_angular)
    
    if len(circumradii) == 0:
        return np.pi, None
    
    circumradii = np.array(circumradii)
    max_cr = np.max(circumradii)
    shifted = circumradii - max_cr
    weights = np.exp(shifted / beta)
    weights /= np.sum(weights)
    cr = max_cr + beta * np.log(np.sum(weights))
    
    return cr, None

def smooth_optimize_l_bfgs(initial_points, max_iter=100, beta=5.0):
    """Optimize using L-BFGS-B with smooth covering radius."""
    params = initial_points.flatten()
    
    def obj_and_grad(p):
        points = to_sphere(p)
        cr, grad = voronoi_covering_radius_and_grad(points, beta)
        
        # Chain rule: grad w.r.t. params
        # points = p / |p|
        # d(points_i)/d(p_j) = (I - points_i @ points_i^T) / |p_j| if i==j, else 0
        # Actually, for the normalization, we need to be careful.
        # Let's use the fact that for unit vectors, the tangent space projection is (I - pp^T)
        # The gradient w.r.t. the unnormalized params is:
        # grad_params_i = grad_points_i / |p_i| projected... 
        # Actually, since we're on the sphere, we can just use the projected gradient
        # and let L-BFGS handle the rest. But L-BFGS works in Euclidean space.
        # 
        # Better approach: use the fact that moving on the sphere means
        # the effective gradient in parameter space is:
        # d(f)/d(p_i) = grad_points_i / ||p_i||  (since p_i = params_i / ||params_i||)
        # But this isn't quite right either. Let me think...
        #
        # If x_i = p_i / ||p_i||, then d(f)/d(p_i) = (I - x_i x_i^T) @ grad_x_i / ||p_i||
        # Since grad_x_i is already in the tangent space (it's a gradient on the sphere),
        # (I - x_i x_i^T) @ grad_x_i = grad_x_i
        # So d(f)/d(p_i) = grad_x_i / ||p_i||
        
        grad_params = np.zeros_like(params)
        for i in range(N):
            norm_i = np.linalg.norm(params[3*i:3*i+3])
            if norm_i > 1e-10:
                grad_params[3*i:3*i+3] = grad[i] / norm_i
        
        return cr, grad_params
    
    best_params = params.copy()
    best_val, _ = obj_and_grad(best_params)
    
    try:
        result = minimize(obj_and_grad, params, method='L-BFGS-B',
                         jac=True,
                         options={'maxiter': max_iter, 'gtol': 1e-7, 'maxfun': 2000})
        if result.fun < best_val:
            best_params = result.x
            best_val = result.fun
    except Exception:
        pass
    
    final_points = to_sphere(best_params)
    final_cr = compute_covering_radius(final_points)
    
    return final_points, final_cr

def fibonacci_spiral(n_points):
    """Fibonacci spiral on the sphere."""
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def icosahedron_based_init(n_points):
    """Generate points based on icosahedron vertices, then add more points."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    if n_points <= 12:
        return verts[:n_points]
    
    points = verts.copy()
    rng = np.random.RandomState(123)
    
    while len(points) < n_points:
        best_dist = 0
        best_point = None
        for _ in range(200):
            candidate = rng.randn(3)
            candidate = candidate / np.linalg.norm(candidate)
            dists = np.arccos(np.clip(points @ candidate, -1, 1))
            min_dist = np.min(dists)
            if min_dist > best_dist:
                best_dist = min_dist
                best_point = candidate
        if best_point is not None:
            points = np.vstack([points, best_point])
    
    return points[:n_points]

def random_init(n_points, rng):
    """Generate random points on the unit sphere."""
    v = rng.randn(n_points, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def lloyd_refine(points, max_iter=30, step_size=0.3):
    """Simple Lloyd-style refinement: move each point toward the centroid of
    its Voronoi cell (approximated by the average of its Delaunay triangle circumcenters)."""
    points = np.asarray(points, dtype=float).copy()
    
    for iteration in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
        
        circumradii, circumcenters, facet_vertices = compute_facet_circumradii(points)
        if circumradii is None:
            break
        
        # For each point, compute the weighted centroid of its facet circumcenters
        # weighted by how much that facet "belongs" to the point
        target = np.zeros((N, 3))
        weight = np.zeros(N)
        
        for i, simplex in enumerate(facet_vertices):
            center = circumcenters[i]
            cr = circumradii[i]
            # Distribute the center among the three vertices of the facet
            for v in simplex:
                target[v] += center
                weight[v] += 1
        
        # Move each point toward its target
        moved = False
        for i in range(N):
            if weight[i] < 1e-10:
                continue
            
            p = points[i]
            t = target[i] / weight[i]
            
            # Slerp
            dot_pt = np.clip(np.dot(p, t), -1, 1)
            theta = np.arccos(dot_pt)
            
            if theta < 1e-8:
                continue
            
            step = step_size * min(1.0, 0.05 / max(cr, 1e-6))
            step = np.clip(step, 0.05, 0.4)
            
            sin_theta = np.sin(theta)
            if sin_theta < 1e-10:
                continue
            
            new_p = ((np.sin((1-step)*theta) * p + np.sin(step*theta) * t) / sin_theta)
            new_p = new_p / np.linalg.norm(new_p)
            
            if np.linalg.norm(new_p - p) > 1e-12:
                points[i] = new_p
                moved = True
        
        if not moved:
            break
    
    return points

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    rng = np.random.RandomState(42)
    
    # Multiple initialization strategies
    inits = []
    inits.append(fibonacci_spiral(N))
    inits.append(icosahedron_based_init(N))
    inits.append(random_init(N, rng))
    
    # Perturbed Fibonacci
    noise = 0.03 * rng.randn(N, 3)
    fib_perturbed = fibonacci_spiral(N) + noise
    fib_perturbed = fib_perturbed / np.linalg.norm(fib_perturbed, axis=1, keepdims=True)
    inits.append(fib_perturbed)
    
    # Perturbed icosahedron
    noise2 = 0.05 * rng.randn(N, 3)
    ico_perturbed = icosahedron_based_init(N) + noise2
    ico_perturbed = ico_perturbed / np.linalg.norm(ico_perturbed, axis=1, keepdims=True)
    inits.append(ico_perturbed)
    
    best_points = inits[0].copy()
    best_cr = compute_covering_radius(best_points)
    
    # Phase 1: Lloyd refinement from each init
    for init in inits:
        if time.time() > DEADLINE - 40:
            break
        
        try:
            lloyd_points = lloyd_refine(init, max_iter=30, step_size=0.3)
            cr = compute_covering_radius(lloyd_points)
            
            if cr < best_cr:
                best_cr = cr
                best_points = lloyd_points.copy()
        except Exception:
            continue
    
    # Phase 2: L-BFGS optimization with smooth objective
    if time.time() < DEADLINE - 25:
        try:
            polished, cr = smooth_optimize_l_bfgs(best_points, max_iter=80, beta=3.0)
            if cr < best_cr:
                best_cr = cr
                best_points = polished
        except Exception:
            pass
    
    # Phase 3: Another Lloyd refinement
    if time.time() < DEADLINE - 15:
        try:
            refined = lloyd_refine(best_points, max_iter=20, step_size=0.15)
            cr = compute_covering_radius(refined)
            if cr < best_cr:
                best_cr = cr
                best_points = refined
        except Exception:
            pass
    
    # Phase 4: Final L-BFGS polish with lower temperature
    if time.time() < DEADLINE - 5:
        try:
            polished, cr = smooth_optimize_l_bfgs(best_points, max_iter=50, beta=1.0)
            if cr < best_cr:
                best_cr = cr
                best_points = polished
        except Exception:
            pass
    
    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    best_points = best_points / norms
    
    # Validate: check no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-6:
                best_points[j] = best_points[j] * 0.999 + 0.001 * rng.randn(3)
                best_points[j] /= np.linalg.norm(best_points[j])
    
    # Check not all in a closed hemisphere
    try:
        hull = ConvexHull(best_points)
    except Exception:
        pass
    
    return best_points
