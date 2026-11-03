"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms


def _covering_radius(points):
    """
    Compute the covering radius in degrees for a set of unit points.
    Uses convex hull facet circumradii.
    """
    if len(points) < 3:
        return 180.0
    try:
        hull = ConvexHull(points)
    except:
        return 180.0
    
    max_angle = 0.0
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        if np.dot(n, p1) < 0:
            n = -n
        
        h = np.dot(n, p1)
        h = np.clip(h, -1.0, 1.0)
        angle = np.arccos(h)
        if angle > max_angle:
            max_angle = angle
            
    return max_angle * 180.0 / np.pi


def _covering_radius_radians(points):
    """Covering radius in radians (for optimization)."""
    return _covering_radius(points) * np.pi / 180.0


def _facet_circumradius_angles(points):
    """Compute circumradius angle for each facet of the convex hull."""
    try:
        hull = ConvexHull(points)
    except:
        return np.array([np.pi/2])
    
    angles = np.zeros(len(hull.simplices))
    for i, simplex in enumerate(hull.simplices):
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            angles[i] = np.pi/2
            continue
        n = n / norm_n
        
        if np.dot(n, p1) < 0:
            n = -n
        
        h = np.dot(n, p1)
        h = np.clip(h, -1.0, 1.0)
        angles[i] = np.arccos(h)
    
    return angles


def _worst_facet_circumcenter(points):
    """
    Find the facet with the largest circumradius and return its circumcenter direction.
    Returns (circumcenter_direction, facet_index, max_angle)
    """
    try:
        hull = ConvexHull(points)
    except:
        return None, -1, 0.0
    
    max_angle = 0.0
    max_facet = -1
    max_cc = None
    
    for i, simplex in enumerate(hull.simplices):
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        if np.dot(n, p1) < 0:
            n = -n
        
        h = np.dot(n, p1)
        h = np.clip(h, -1.0, 1.0)
        angle = np.arccos(h)
        
        if angle > max_angle:
            max_angle = angle
            max_facet = i
            max_cc = n
    
    return max_cc, max_facet, max_angle


def _worst_facet_shrinkage(points, time_limit=30, damping=0.3, max_iter=500):
    """
    Deterministic iterative method: identify the worst facet (largest circumradius)
    and move its three vertices toward the facet's circumcenter along great circles.
    This directly shrinks the largest empty cap.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    best_points = points.copy()
    best_cost = _covering_radius(points)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        cc, facet_idx, max_angle = _worst_facet_circumcenter(points)
        if cc is None or facet_idx < 0:
            break
        
        # Check if we've converged
        if max_angle * 180.0 / np.pi < 17.5:
            break
        
        # Get the three vertices of the worst facet
        try:
            hull = ConvexHull(points)
            simplex = hull.simplices[facet_idx]
        except:
            break
        
        v0, v1, v2 = simplex
        
        # Move each vertex toward the circumcenter
        new_points = points.copy()
        for vi in [v0, v1, v2]:
            p = points[vi]
            # Great circle interpolation toward cc
            # p + damping * (cc - p), then normalize
            new_p = p + damping * (cc - p)
            norm_new = np.linalg.norm(new_p)
            if norm_new > 1e-10:
                new_points[vi] = new_p / norm_new
        
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_points = new_points.copy()
            points = new_points
        else:
            # Try smaller damping
            new_points2 = points.copy()
            for vi in [v0, v1, v2]:
                p = points[vi]
                new_p = p + damping * 0.3 * (cc - p)
                norm_new = np.linalg.norm(new_p)
                if norm_new > 1e-10:
                    new_points2[vi] = new_p / norm_new
            new_points2 = _normalize(new_points2)
            new_cost2 = _covering_radius(new_points2)
            if new_cost2 < best_cost:
                best_cost = new_cost2
                best_points = new_points2.copy()
                points = new_points2
            else:
                # Try even smaller
                new_points3 = points.copy()
                for vi in [v0, v1, v2]:
                    p = points[vi]
                    new_p = p + damping * 0.1 * (cc - p)
                    norm_new = np.linalg.norm(new_p)
                    if norm_new > 1e-10:
                        new_points3[vi] = new_p / norm_new
                new_points3 = _normalize(new_points3)
                new_cost3 = _covering_radius(new_points3)
                if new_cost3 < best_cost:
                    best_cost = new_cost3
                    best_points = new_points3.copy()
                    points = new_points3
                else:
                    break
    
    return best_points


def _multi_worst_facet_shrinkage(points, time_limit=30, damping=0.2, max_iter=300, top_k=3):
    """
    Move vertices of the top-k worst facets toward their circumcenters simultaneously.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    best_points = points.copy()
    best_cost = _covering_radius(points)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
        
        # Compute all facet angles
        angles = []
        for i, simplex in enumerate(hull.simplices):
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                angles.append((0.0, i, None))
                continue
            n = n / norm_n
            if np.dot(n, p1) < 0:
                n = -n
            h = np.dot(n, p1)
            h = np.clip(h, -1.0, 1.0)
            angle = np.arccos(h)
            angles.append((angle, i, n))
        
        # Sort by angle descending
        angles.sort(key=lambda x: -x[0])
        
        if not angles or angles[0][0] * 180.0 / np.pi < 17.5:
            break
        
        # Take top-k facets
        new_points = points.copy()
        moved = set()
        
        for angle, idx, cc in angles[:top_k]:
            if cc is None:
                continue
            simplex = hull.simplices[idx]
            for vi in simplex:
                if vi not in moved:
                    p = points[vi]
                    new_p = p + damping * (cc - p)
                    norm_new = np.linalg.norm(new_p)
                    if norm_new > 1e-10:
                        new_points[vi] = new_p / norm_new
                    moved.add(vi)
        
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_points = new_points.copy()
            points = new_points
        else:
            # Try smaller damping
            new_points2 = points.copy()
            moved2 = set()
            for angle, idx, cc in angles[:top_k]:
                if cc is None:
                    continue
                simplex = hull.simplices[idx]
                for vi in simplex:
                    if vi not in moved2:
                        p = points[vi]
                        new_p = p + damping * 0.3 * (cc - p)
                        norm_new = np.linalg.norm(new_p)
                        if norm_new > 1e-10:
                            new_points2[vi] = new_p / norm_new
                        moved2.add(vi)
            
            new_points2 = _normalize(new_points2)
            new_cost2 = _covering_radius(new_points2)
            if new_cost2 < best_cost:
                best_cost = new_cost2
                best_points = new_points2.copy()
                points = new_points2
            else:
                break
    
    return best_points


def _spherical_lloyd_iteration(points, max_iter=100, time_limit=30, damping=0.5):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    Moves points towards the circumcenter of their largest adjacent facet.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
        
        new_points = points.copy()
        
        # Find the largest facet for each point
        point_to_facets = {i: [] for i in range(N)}
        for fi, simplex in enumerate(hull.simplices):
            for pi in simplex:
                point_to_facets[pi].append(fi)
        
        # Compute facet circumradius angles
        facet_angles = _facet_circumradius_angles(points)
        
        for pi in range(N):
            facets = point_to_facets[pi]
            if not facets:
                continue
            
            # Find the largest facet adjacent to this point
            max_angle_idx = max(facets, key=lambda fi: facet_angles[fi])
            simplex = hull.simplices[max_angle_idx]
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            n = n / norm_n
            
            if np.dot(n, p1) < 0:
                n = -n
            
            # Move towards the circumcenter (normal direction)
            new_points[pi] = points[pi] + damping * (n - points[pi])
            new_points[pi] = new_points[pi] / np.linalg.norm(new_points[pi])
        
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
        else:
            # Try with smaller damping
            new_points2 = points.copy()
            for pi in range(N):
                facets = point_to_facets[pi]
                if not facets:
                    continue
                max_angle_idx = max(facets, key=lambda fi: facet_angles[fi])
                simplex = hull.simplices[max_angle_idx]
                p1 = points[simplex[0]]
                p2 = points[simplex[1]]
                p3 = points[simplex[2]]
                
                v1 = p2 - p1
                v2 = p3 - p1
                n = np.cross(v1, v2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                n = n / norm_n
                
                if np.dot(n, p1) < 0:
                    n = -n
                
                new_points2[pi] = points[pi] + damping * 0.3 * (n - points[pi])
                new_points2[pi] = new_points2[pi] / np.linalg.norm(new_points2[pi])
            
            new_points2 = _normalize(new_points2)
            new_cost2 = _covering_radius(new_points2)
            
            if new_cost2 < current_cost:
                current_cost = new_cost2
                points = new_points2
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_points = points.copy()
            else:
                break
    
    return best_points


def _nelder_mead_optimize(points, time_limit=20, maxiter=2000):
    """
    Local optimization using Nelder-Mead on the true covering radius.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        return _covering_radius_radians(pts)
    
    x0 = points.flatten()
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': maxiter, 'xatol': 1e-14, 'fatol': 1e-16, 
                               'adaptive': True, 'maxls': 50})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _facet_smooth_surrogate(params, beta=50.0):
    """
    Smooth surrogate based on convex hull facet circumradii.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    angles = _facet_circumradius_angles(pts)
    
    beta_angles = beta * angles
    max_val = np.max(beta_angles)
    logsumexp_max = max_val + np.log(np.sum(np.exp(beta_angles - max_val)))
    soft_max = logsumexp_max / beta
    
    return soft_max


def _smooth_gradient_optimize(points, time_limit=30, beta=100.0):
    """
    Gradient-based optimization using smooth surrogate with L-BFGS-B.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        return _facet_smooth_surrogate(params, beta=beta)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 3000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _fibonacci_points():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(seed=42):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(N, 3)
    return _normalize(points)


def _icosahedron_based_points():
    """Generate points based on icosahedron vertices with perturbations."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = _normalize(verts)
    
    rng = np.random.RandomState(123)
    points = [verts[i] for i in range(12)]
    
    while len(points) < N:
        base = verts[rng.randint(12)]
        noise = rng.randn(3) * 0.3
        new_pt = base + noise
        new_pt = new_pt / np.linalg.norm(new_pt)
        min_dist = min(np.linalg.norm(new_pt - p) for p in points)
        if min_dist > 0.1:
            points.append(new_pt)
    
    return np.array(points[:N])


def _inverse_repulsion_optimize(points, time_limit=30, max_iter=200, k=4, temperature=0.1):
    """
    Symplectic-like repulsion: points repel each other with inverse-k power,
    with a slight attraction toward uniform distribution.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    best_points = points.copy()
    best_cost = _covering_radius(points)
    
    velocities = np.zeros((N, 3))
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute repulsive forces
        forces = np.zeros((N, 3))
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                # Inverse power repulsion
                force_mag = k / (dist ** 3)
                forces[i] += force_mag * diff / dist
        
        # Add small random perturbation to escape local minima
        if it % 10 == 0:
            rng = np.random.RandomState(it)
            forces += temperature * rng.randn(N, 3)
        
        # Update positions (simple Euler step with damping)
        damping = 0.95
        velocities = damping * velocities + 0.01 * forces
        new_points = points + velocities
        new_points = _normalize(new_points)
        
        new_cost = _covering_radius(new_points)
        if new_cost < best_cost:
            best_cost = new_cost
            best_points = new_points.copy()
            points = new_points
        else:
            # Revert and reduce step
            velocities *= 0.5
            new_points = points + velocities
            new_points = _normalize(new_points)
            new_cost = _covering_radius(new_points)
            if new_cost < best_cost:
                best_cost = new_cost
                best_points = new_points.copy()
                points = new_points
    
    return best_points


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 145.0  # Safety margin for 150s limit
    start_time = time.time()
    
    def _time_left():
        return total_time_limit - (time.time() - start_time)
    
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_points())
    
    # Random with different seeds
    for seed in [42, 123, 777, 999]:
        initializations.append(_random_points(seed=seed))
    
    # Icosahedron-based
    initializations.append(_icosahedron_based_points())
    
    best_points = None
    best_cost = np.inf
    
    # Phase 1: Worst-facet shrinkage from multiple starts (primary method)
    # This is the key new deterministic geometric method
    shrink_time_each = 8.0
    for init_pts in initializations:
        if _time_left() < 40:
            break
        
        pts = _worst_facet_shrinkage(init_pts, time_limit=min(shrink_time_each, _time_left() - 35), damping=0.3, max_iter=500)
        cost = _covering_radius(pts)
        
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
        best_cost = _covering_radius(best_points)
    
    # Phase 2: Multi-worst-facet shrinkage refinement
    if _time_left() > 50:
        pts = _multi_worst_facet_shrinkage(best_points, time_limit=min(20, _time_left() - 40), damping=0.15, max_iter=300, top_k=3)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 3: Lloyd's relaxation for global balance
    if _time_left() > 40:
        pts = _spherical_lloyd_iteration(best_points, max_iter=100, time_limit=min(15, _time_left() - 30), damping=0.3)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 4: Another round of worst-facet shrinkage
    if _time_left() > 30:
        pts = _worst_facet_shrinkage(best_points, time_limit=min(15, _time_left() - 25), damping=0.15, max_iter=500)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 5: Inverse repulsion for fine-tuning
    if _time_left() > 25:
        pts = _inverse_repulsion_optimize(best_points, time_limit=min(12, _time_left() - 20), max_iter=200, k=4, temperature=0.05)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 6: Final Nelder-Mead polish
    if _time_left() > 15:
        pts = _nelder_mead_optimize(best_points, time_limit=min(10, _time_left() - 10), maxiter=3000)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 7: One more worst-facet pass
    if _time_left() > 8:
        pts = _worst_facet_shrinkage(best_points, time_limit=min(5, _time_left() - 3), damping=0.05, max_iter=200)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
