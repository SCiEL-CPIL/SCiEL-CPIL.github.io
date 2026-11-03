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


def _active_facets(points, threshold=0.001):
    """
    Identify active facets (those within threshold of max circumradius).
    Returns list of simplex indices.
    """
    angles = _facet_circumradius_angles(points)
    max_angle = np.max(angles)
    active = np.where(angles >= max_angle - threshold)[0]
    return active, max_angle


def _force_based_relaxation(points, time_limit=10.0, target_radius=None,
                             repulsion_strength=1.0, attraction_strength=1.0,
                             max_iter=500, cooling=0.999):
    """
    Force-based relaxation: repulsion between points + attraction to gap centers.
    Points are pushed apart (repulsion) and pulled toward the center of their
    largest adjacent facet (gap-filling).
    """
    start_time = time.time()
    points = _normalize(points).copy()
    
    if target_radius is None:
        target_radius = _covering_radius_radians(points)
    
    best_points = points.copy()
    best_cost = _covering_radius_radians(points)
    
    temperature = 0.05
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # 1. Repulsion: push points apart to maintain minimum separation
        for i in range(N):
            for j in range(i+1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random perturbation for coincident points
                    forces[i] += np.random.randn(3) * 0.1
                    forces[j] -= np.random.randn(3) * 0.1
                    continue
                # Repulsive force (inverse square, capped)
                force_mag = repulsion_strength / (dist * dist + 0.01)
                force_mag = min(force_mag, 5.0)  # Cap force
                forces[i] += force_mag * diff / dist
                forces[j] -= force_mag * diff / dist
        
        # 2. Gap-filling attraction: pull points toward center of largest adjacent facet
        facet_angles = _facet_circumradius_angles(points)
        max_angle = np.max(facet_angles)
        
        # For each point, find its largest adjacent facet and pull toward its center
        point_to_facets = {i: [] for i in range(N)}
        for fi, simplex in enumerate(hull.simplices):
            for pi in simplex:
                point_to_facets[pi].append(fi)
        
        for pi in range(N):
            facets = point_to_facets[pi]
            if not facets:
                continue
            
            # Weight by how close each facet is to the max
            weights = []
            for fi in facets:
                # Exponential weight: facets closer to max get more weight
                w = np.exp(10.0 * (facet_angles[fi] - max_angle))
                weights.append(w)
            
            total_w = sum(weights)
            if total_w < 1e-10:
                continue
            
            # Compute weighted center of adjacent facets
            center = np.zeros(3)
            for fi, w in zip(facets, weights):
                simplex = hull.simplices[fi]
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
                
                center += w * n
            
            center = center / total_w
            center_norm = np.linalg.norm(center)
            if center_norm > 1e-10:
                center = center / center_norm
            
            # Attraction force toward gap center
            diff = center - points[pi]
            forces[pi] += attraction_strength * diff
        
        # Apply forces with cooling
        step_size = temperature * (1.0 - iteration / max_iter)
        new_points = points + step_size * forces
        new_points = _normalize(new_points)
        
        # Check improvement
        new_cost = _covering_radius_radians(new_points)
        if new_cost < best_cost:
            best_cost = new_cost
            best_points = new_points.copy()
        
        # Accept if improved or with probability (simulated annealing)
        if new_cost <= _covering_radius_radians(points) or \
           np.random.random() < np.exp(-(new_cost - _covering_radius_radians(points)) / temperature):
            points = new_points
        
        temperature *= cooling
    
    return best_points


def _epigraph_surrogate(params, beta=50.0):
    """
    Smooth epigraph formulation: minimize t subject to all facet angles <= t.
    Uses log-sum-exp to smoothly approximate the max.
    Returns (t, smooth_max) where smooth_max approximates max(facet_angles).
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    angles = _facet_circumradius_angles(pts)
    
    # Log-sum-exp approximation of max
    beta_angles = beta * angles
    max_val = np.max(beta_angles)
    logsumexp = max_val + np.log(np.sum(np.exp(beta_angles - max_val)))
    smooth_max = logsumexp / beta
    
    return smooth_max


def _epigraph_optimize(points, time_limit=20.0, beta=80.0, maxiter=2000):
    """
    Optimize using epigraph formulation with L-BFGS-B.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        return _epigraph_surrogate(params, beta=beta)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-14, 'maxfun': 5000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _active_facet_polish(points, time_limit=15.0, max_iter=1000):
    """
    High-precision polish: identify active facets and solve for equal circumradius.
    Uses Nelder-Mead on only the active facet constraints.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        
        angles = _facet_circumradius_angles(pts)
        max_angle = np.max(angles)
        
        # Penalize variance of active facets + max angle
        active = angles >= max_angle - 0.005
        if np.sum(active) > 1:
            active_angles = angles[active]
            variance = np.var(active_angles)
        else:
            variance = 0.0
        
        return max_angle + 0.1 * variance
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-15, 'fatol': 1e-16,
                               'adaptive': True, 'maxls': 100})
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


def _jittered_fibonacci_points(seed=0):
    """Fibonacci points with small random jitter."""
    pts = _fibonacci_points()
    rng = np.random.RandomState(seed)
    jitter = rng.randn(N, 3) * 0.05
    pts = pts + jitter
    return _normalize(pts)


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 145.0  # Safety margin for 150s limit
    start_time = time.time()
    
    def _time_left():
        return total_time_limit - (time.time() - start_time)
    
    # Multiple diverse initializations
    initializations = []
    initializations.append(_fibonacci_points())
    initializations.append(_jittered_fibonacci_points(seed=1))
    initializations.append(_jittered_fibonacci_points(seed=2))
    for seed in [42, 123, 777, 999, 555, 888]:
        initializations.append(_random_points(seed=seed))
    initializations.append(_icosahedron_based_points())
    
    best_points = None
    best_cost = np.inf
    
    # Phase 1: Force-based relaxation from multiple starts
    # This is the most effective strategy per cell findings
    force_time_each = 8.0
    for init_pts in initializations:
        if _time_left() < 40:  # Keep reserve for later phases
            break
        
        pts = _force_based_relaxation(init_pts, time_limit=min(force_time_each, _time_left() - 35),
                                       repulsion_strength=0.8, attraction_strength=1.2,
                                       max_iter=300)
        cost = _covering_radius_radians(pts)
        
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
        best_cost = _covering_radius_radians(best_points)
    
    # Phase 2: Epigraph smooth optimization
    if _time_left() > 50:
        pts = _epigraph_optimize(best_points, time_limit=min(25, _time_left() - 40), beta=60.0, maxiter=3000)
        cost = _covering_radius_radians(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
        
        if _time_left() > 35:
            pts = _epigraph_optimize(best_points, time_limit=min(15, _time_left() - 30), beta=120.0, maxiter=2000)
            cost = _covering_radius_radians(pts)
            if cost < best_cost:
                best_cost = cost
                best_points = pts.copy()
    
    # Phase 3: Force-based refinement from current best
    if _time_left() > 30:
        pts = _force_based_relaxation(best_points, time_limit=min(15, _time_left() - 25),
                                       repulsion_strength=0.5, attraction_strength=1.5,
                                       max_iter=200)
        cost = _covering_radius_radians(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 4: Active facet polish
    if _time_left() > 20:
        pts = _active_facet_polish(best_points, time_limit=min(15, _time_left() - 15), max_iter=3000)
        cost = _covering_radius_radians(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 5: Final Nelder-Mead on exact objective
    if _time_left() > 15:
        def final_objective(params):
            pts = params.reshape(N, 3)
            pts = _normalize(pts)
            return _covering_radius_radians(pts)
        
        x0 = best_points.flatten()
        result = minimize(final_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-16,
                                   'adaptive': True, 'maxls': 100})
        pts = result.x.reshape(N, 3)
        pts = _normalize(pts)
        cost = _covering_radius_radians(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts
    
    # Phase 6: One more force-based pass if time allows
    if _time_left() > 10:
        pts = _force_based_relaxation(best_points, time_limit=min(5, _time_left() - 5),
                                       repulsion_strength=0.3, attraction_strength=1.0,
                                       max_iter=100)
        cost = _covering_radius_radians(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts
    
    best_points = _normalize(best_points)
    
    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_points()
    
    return best_points
# EVOLVE-BLOCK-END
