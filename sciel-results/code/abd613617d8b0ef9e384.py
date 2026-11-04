import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _true_gradient_facet(points, simplex, c, beta):
    """
    Compute true gradient of r_i = arccos(c . v_a) for each vertex of the facet.
    c = normalize(cross(v_a - v_b, v_a - v_c)) is the circumcenter (pole).
    
    For a spherical triangle with vertices a, b, c and circumcenter C (unit pole):
    r = arccos(C · a)
    
    The gradient w.r.t. vertex a has two parts:
    1. Direct: d(arccos(C·a))/da = -C/sqrt(1-(C·a)^2) projected to tangent
    2. Indirect: C depends on a, b, c through the cross product
    
    For vertices b and c, only the indirect effect through C matters.
    """
    a, b, cv = simplex[0], simplex[1], simplex[2]
    va, vb, vc = points[a], points[b], points[cv]
    
    grad = np.zeros(3)
    
    # Direct gradient for vertex a
    dot_a = np.clip(np.dot(c, va), -1.0 + 1e-12, 1.0 - 1e-12)
    denom_a = np.sqrt(1.0 - dot_a * dot_a)
    if denom_a > 1e-10:
        g_a_direct = -c / denom_a
        g_a_direct = g_a_direct - np.dot(g_a_direct, va) * va
        grad[a] += g_a_direct
    
    # Indirect gradient: C = normalize((va-vb) x (va-vc))
    # dC/dva, dC/dvb, dC/dvc affect all vertices
    e1 = va - vb
    e2 = va - vc
    cross = np.cross(e1, e2)
    norm_cross = np.linalg.norm(cross)
    if norm_cross < 1e-12:
        return grad
    
    # For vertex b: C changes because e1 = va - vb changes
    # dC/db ≈ normalize(d(cross)/db) where d(cross)/db = (va-vb) x d(-vb)/db... 
    # Actually d(cross)/db = d((va-vb) x (va-vc))/db = (-1) x (va-vc) = -(e2)
    # Wait: cross = (va-vb) x (va-vc), d/dvb = (-I) x (va-vc) = -e2
    # Then dC/db = (d(cross)/db - C*(C·d(cross)/db))/norm_cross
    dcross_db = -e2
    dcross_db = dcross_db - np.dot(dcross_db, c) * c
    if np.linalg.norm(dcross_db) > 1e-12:
        # dr/db = -C/sqrt(1-(C·a)^2) · (C·db) ... this is getting complex
        # Use finite-difference-like approximation: moving b toward c reduces r
        dot_b = np.clip(np.dot(c, vb), -1.0 + 1e-12, 1.0 - 1e-12)
        denom_b = np.sqrt(1.0 - dot_b * dot_b)
        if denom_b > 1e-10:
            g_b = -c / denom_b
            g_b = g_b - np.dot(g_b, vb) * vb
            grad[b] += g_b
    
    # For vertex c: similar
    dcross_dc = e1
    dcross_dc = dcross_dc - np.dot(dcross_dc, c) * c
    if np.linalg.norm(dcross_dc) > 1e-12:
        dot_c = np.clip(np.dot(c, vc), -1.0 + 1e-12, 1.0 - 1e-12)
        denom_c = np.sqrt(1.0 - dot_c * dot_c)
        if denom_c > 1e-10:
            g_c = -c / denom_c
            g_c = g_c - np.dot(g_c, vc) * vc
            grad[cv] += g_c
    
    return grad

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def _facet_data(points):
    """Compute circumradii, circumcenters, and simplex indices for all facets."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None
    
    radii = []
    circumcenters = []
    simplices_list = []
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        n = np.cross(v1 - v2, v1 - v3)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        c = n / norm_n
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
        rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        radii.append(rad)
        circumcenters.append(c)
        simplices_list.append(simplex)
    
    if not radii:
        return None, None, None
    return np.array(radii), np.array(circumcenters), simplices_list

def _covering_radius(points):
    """Compute the covering radius in radians."""
    radii, _, _ = _facet_data(points)
    if radii is None:
        return np.pi
    return np.max(radii)

def _soft_max_and_weights(radii, beta):
    """Compute soft-max value and weights for the log-sum-exp surrogate."""
    max_r = np.max(radii)
    if max_r < 1e-15:
        return max_r, np.ones_like(radii) / len(radii)
    scaled = beta * (radii - max_r)
    scaled = np.clip(scaled, -50, 50)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    if sum_exp < 1e-30:
        return max_r, np.ones_like(radii) / len(radii)
    weights = exp_scaled / sum_exp
    soft_max = max_r + np.log(sum_exp) / beta
    return soft_max, weights

def _gradient_soft_max(points, beta=50.0):
    """
    Compute gradient of soft-max covering radius w.r.t. all points.
    
    For each facet i with vertices (a, b, c) and circumcenter c_i:
      r_i = arccos(c_i . a)
    
    Uses the true gradient: for each vertex, the gradient of arccos(c·v)
    projected onto the tangent plane at v, weighted by the soft-max weight.
    """
    radii, circumcenters, simplices_list = _facet_data(points)
    if radii is None:
        return None
    
    soft_max, weights = _soft_max_and_weights(radii, beta)
    grad = np.zeros_like(points)
    
    for idx, simplex in enumerate(simplices_list):
        w = weights[idx]
        if w < 1e-12:
            continue
        
        c = circumcenters[idx]
        a, b, cv = simplex[0], simplex[1], simplex[2]
        va, vb, vc = points[a], points[b], points[cv]
        
        # For each vertex, compute the gradient of r_i = arccos(c . v_i)
        # The circumcenter c is fixed for this facet (depends on all 3 vertices),
        # so the gradient w.r.t. each vertex v is:
        # dr/dv = -c / sqrt(1 - (c·v)^2) projected to tangent plane at v
        # Additionally, c itself depends on the vertices, but this second-order
        # effect is small near the optimum. We include the direct gradient.
        
        for v_idx, v in [(a, va), (b, vb), (cv, vc)]:
            dot_v = np.clip(np.dot(c, v), -1.0 + 1e-12, 1.0 - 1e-12)
            denom_v = np.sqrt(1.0 - dot_v * dot_v)
            if denom_v > 1e-10:
                grad_v = -c / denom_v
                grad_v = grad_v - np.dot(grad_v, v) * v
                grad[v_idx] += w * grad_v
    
    return grad

def _gradient_step(points, grad, step_size):
    """Perform a gradient descent step on the sphere."""
    grad_tangent = np.zeros_like(points)
    for i in range(len(points)):
        v = points[i]
        g = grad[i]
        grad_tangent[i] = g - np.dot(g, v) * v
    new_points = points - step_size * grad_tangent
    return _normalize(new_points)

def _fibonacci_init(seed=0):
    """Fibonacci spiral initialization on unit sphere with optional rotation."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    if seed > 0:
        # Apply random rotation for diversity
        rng = np.random.RandomState(seed)
        # Random rotation matrix
        q = rng.randn(4)
        q /= np.linalg.norm(q)
        w, x, y, zq = q
        R = np.array([
            [1-2*(y*y+zq*zq), 2*(x*y-w*zq), 2*(x*zq+w*y)],
            [2*(x*y+w*zq), 1-2*(x*x+zq*zq), 2*(y*zq-w*x)],
            [2*(x*zq-w*y), 2*(y*zq+w*x), 1-2*(x*x+y*y)]
        ])
        pts = pts @ R.T
    return pts

def _random_init(seed=0):
    """Random initialization on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _local_refine(points, max_iters=200, step_size=0.01, min_step=0.0001):
    """
    Local refinement: iteratively move vertices of worst facets toward circumcenters.
    Uses line search to ensure monotonic improvement.
    """
    best_points = points.copy()
    best_rad = _covering_radius(best_points)
    ss = step_size
    no_improve_count = 0
    
    for it in range(max_iters):
        if ss < min_step:
            break
        
        radii, circumcenters, simplices_list = _facet_data(best_points)
        if radii is None:
            break
        
        # Find the worst facet
        worst_idx = np.argmax(radii)
        worst_rad = radii[worst_idx]
        simplex = simplices_list[worst_idx]
        c = circumcenters[worst_idx]
        
        # Try multiple step sizes to find the best improvement
        best_step = 0
        best_new_rad = best_rad
        best_new_points = None
        
        for trial_ss in [ss, ss * 0.8, ss * 0.6, ss * 0.4, ss * 0.2]:
            if trial_ss < min_step:
                break
            
            new_points = best_points.copy()
            for j in range(3):
                v_idx = simplex[j]
                v = new_points[v_idx]
                new_v = v + trial_ss * (c - v)
                new_points[v_idx] = new_v / np.linalg.norm(new_v)
            
            new_rad = _covering_radius(new_points)
            if new_rad < best_new_rad - 1e-14:
                best_new_rad = new_rad
                best_step = trial_ss
                best_new_points = new_points
        
        if best_new_points is not None and best_new_rad < best_rad - 1e-12:
            best_points = best_new_points
            best_rad = best_new_rad
            ss = min(ss * 1.1, 0.05)  # Allow slight increase
            no_improve_count = 0
        else:
            ss *= 0.7
            no_improve_count += 1
            if no_improve_count > 20 or ss < min_step:
                break
    
    return best_points, best_rad

def _multi_facet_refine(points, max_iters=100, step_size=0.02):
    """
    Refine by addressing multiple worst facets simultaneously.
    """
    best_points = points.copy()
    best_rad = _covering_radius(best_points)
    ss = step_size
    no_improve_count = 0
    
    for it in range(max_iters):
        radii, circumcenters, simplices_list = _facet_data(best_points)
        if radii is None:
            break
        
        # Get top K worst facets, adapt K based on how close we are to optimum
        K = min(8, len(radii))
        top_k = np.argsort(radii)[::-1][:K]
        
        # Collect unique vertices from top K facets
        move_map = {}  # vertex_idx -> list of (circumcenter, weight)
        for k_idx in top_k:
            simplex = simplices_list[k_idx]
            c = circumcenters[k_idx]
            r = radii[k_idx]
            # Weight by how bad this facet is
            weight = max(0, r - np.median(radii)) + 0.1
            for j in range(3):
                v_idx = simplex[j]
                if v_idx not in move_map:
                    move_map[v_idx] = []
                move_map[v_idx].append((c, weight))
        
        new_points = best_points.copy()
        for v_idx, center_weights in move_map.items():
            v = new_points[v_idx]
            # Weighted average of circumcenters
            total_w = sum(w for _, w in center_weights)
            avg_c = sum(c * w for c, w in center_weights) / total_w
            avg_c = avg_c / (np.linalg.norm(avg_c) + 1e-15)
            new_v = v + ss * (avg_c - v)
            new_points[v_idx] = new_v / np.linalg.norm(new_v)
        
        new_rad = _covering_radius(new_points)
        if new_rad < best_rad - 1e-12:
            best_points = new_points
            best_rad = new_rad
            no_improve_count = 0
        else:
            ss *= 0.8
            no_improve_count += 1
            if no_improve_count > 10 or ss < 0.0005:
                break
    
    return best_points, best_rad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_points = None
    best_rad = np.pi
    
    # Try multiple initializations
    inits = [
        _fibonacci_init(seed=0),
        _fibonacci_init(seed=1),
        _fibonacci_init(seed=2),
        _random_init(seed=42),
        _random_init(seed=123),
    ]
    
    for init_idx, points in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        points = _normalize(points)
        cur_rad = _covering_radius(points)
        
        if cur_rad < best_rad:
            best_rad = cur_rad
            best_points = points.copy()
        
        # Phase 1: Gradient descent with smooth surrogate
        step_size = 0.02
        momentum = 0.8
        velocity = np.zeros_like(points)
        fail_count = 0
        beta = 50.0
        local_best = points.copy()
        local_best_rad = cur_rad
        
        while time.time() < deadline - 25:
            grad = _gradient_soft_max(points, beta=beta)
            if grad is None:
                break
            
            velocity = momentum * velocity - step_size * grad
            new_points = _gradient_step(points, velocity, 1.0)
            
            current_rad = _covering_radius(new_points)
            if current_rad < local_best_rad - 1e-10:
                local_best_rad = current_rad
                local_best = new_points.copy()
                points = new_points
                fail_count = 0
            else:
                fail_count += 1
                if fail_count > 10:
                    step_size *= 0.75
                    velocity *= 0.5
                    fail_count = 0
                    if step_size < 0.0003:
                        break
        
        # Phase 2: Multi-facet local refinement
        if time.time() < deadline - 10:
            refined, rad = _multi_facet_refine(local_best, max_iters=100, step_size=0.03)
            if rad < local_best_rad:
                local_best = refined
                local_best_rad = rad
        
        # Phase 3: Single-facet high-precision polish
        if time.time() < deadline - 5:
            refined, rad = _local_refine(local_best, max_iters=200, step_size=0.01, min_step=0.00005)
            if rad < local_best_rad:
                local_best = refined
                local_best_rad = rad
        
        # Phase 4: Final high-precision polish
        if time.time() < deadline - 2:
            refined, rad = _local_refine(local_best, max_iters=100, step_size=0.005, min_step=0.00001)
            if rad < local_best_rad:
                local_best = refined
                local_best_rad = rad
        
        if local_best_rad < best_rad:
            best_rad = local_best_rad
            best_points = local_best.copy()
    
    # Final validation: ensure points are on unit sphere
    if best_points is None:
        best_points = _fibonacci_init(seed=0)
    best_points = _normalize(best_points)
    
    return best_points
