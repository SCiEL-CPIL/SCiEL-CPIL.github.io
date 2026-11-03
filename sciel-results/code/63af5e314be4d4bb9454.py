import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return pts / norms

def _fibonacci_init(seed=0):
    """Generate n points on unit sphere using Fibonacci spiral with jitter."""
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2 * np.pi)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Add small random perturbation to break symmetry
    for idx in range(N):
        p = pts[idx]
        rand_vec = rng.randn(3)
        tangent = rand_vec - np.dot(rand_vec, p) * p
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-10:
            pts[idx] = p + 0.01 * tangent / tangent_norm
            pts[idx] = pts[idx] / np.linalg.norm(pts[idx])
    return _normalize(pts)

def _circumcenter_robust(p1, p2, p3):
    """
    Compute the circumcenter of a spherical triangle (3 points on unit sphere).
    Returns the point on the unit sphere equidistant from all three vertices.
    This is the center of the circumscribed spherical cap.
    """
    # The circumcenter of a spherical triangle with vertices a, b, c
    # is proportional to a*sin(A) + b*sin(B) + c*sin(C)
    # where A, B, C are the angles at vertices a, b, c.
    
    def ang_dist(p, q):
        dot = np.clip(np.dot(p, q), -1.0, 1.0)
        return np.arccos(dot)
    
    A = ang_dist(p2, p3)  # angle at p1
    B = ang_dist(p1, p3)  # angle at p2
    C = ang_dist(p1, p2)  # angle at p3
    
    center = p1 * np.sin(A) + p2 * np.sin(B) + p3 * np.sin(C)
    norm = np.linalg.norm(center)
    if norm < 1e-10:
        # Degenerate case: use centroid
        center = (p1 + p2 + p3) / 3.0
        norm = np.linalg.norm(center)
    if norm < 1e-10:
        return p1.copy()
    return center / norm

def _compute_circumradii(pts, hull):
    """
    Compute angular circumradii for all triangular facets of the convex hull.
    Returns array of circumradii (in radians) and the circumcenters.
    """
    simplices = hull.simplices
    n_facets = len(simplices)
    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    
    for k, tri in enumerate(simplices):
        v0 = pts[tri[0]]
        v1 = pts[tri[1]]
        v2 = pts[tri[2]]
        
        c = _circumcenter_robust(v0, v1, v2)
        centers[k] = c
        
        dot = np.clip(np.dot(c, v0), -1.0, 1.0)
        radii[k] = np.arccos(dot)
        
    return radii, centers

def _covering_radius(pts):
    """
    Compute the covering radius of a set of points on the unit sphere.
    This is the maximum circumradius of all facets of the convex hull.
    Returns (radius_in_degrees, radii_array, centers_array) or (180.0, None, None) on failure.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0, None, None
    
    radii, centers = _compute_circumradii(pts, hull)
    max_radius = np.max(radii)
    return np.degrees(max_radius), radii, centers

def _repulsive_force(pts):
    """
    Compute repulsive forces between points on the sphere.
    Points repel each other with force proportional to 1/r^2 (Coulomb-like).
    The force is projected onto the tangent space of the sphere.
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Points are coincident: add random perturbation
                rand_vec = np.random.randn(3)
                forces[i] += rand_vec
                forces[j] -= rand_vec
                continue
            
            # Coulomb force: F = k * diff / dist^3
            force_mag = 1.0 / (dist * dist)
            force_dir = diff / dist
            force = force_mag * force_dir
            
            forces[i] += force
            forces[j] -= force
    
    # Project forces onto tangent space
    for i in range(n):
        p = pts[i]
        f = forces[i]
        forces[i] = f - np.dot(f, p) * p
    
    return forces

def _opti_cov(pts, deadline):
    """
    Optimize points to minimize the covering radius using a combination of:
    1. Repulsive force simulation to get a good initial distribution
    2. Lloyd step: move each point towards centroid of adjacent facet circumcenters
    3. Greedy step: shrink the largest facet
    4. Nelder-Mead polish on top facets
    """
    current_pts = pts.copy()
    best_pts = pts.copy()
    best_radius, _, _ = _covering_radius(current_pts)
    
    # Pre-compute initial hull
    try:
        hull = ConvexHull(current_pts)
        radii, centers = _compute_circumradii(current_pts, hull)
    except Exception:
        return best_pts
    
    step_size_greedy = 0.05
    step_size_lloyd = 0.02
    no_improve_count = 0
    max_no_improve = 30
    iteration = 0
    
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # Alternate between Lloyd and Greedy
        use_lloyd = (iteration % 3 == 0)
        
        if use_lloyd:
            # Lloyd step: move each point towards centroid of adjacent facet circumcenters
            simplices = hull.simplices
            point_to_facets = [[] for _ in range(N)]
            for k, tri in enumerate(simplices):
                for idx in tri:
                    point_to_facets[idx].append(k)
                    
            new_pts = current_pts.copy()
            for i in range(N):
                facets = point_to_facets[i]
                if not facets:
                    continue
                    
                sum_c = np.sum(centers[facets], axis=0)
                norm_c = np.linalg.norm(sum_c)
                if norm_c < 1e-10:
                    continue
                    
                target = sum_c / norm_c
                new_pts[i] = current_pts[i] + step_size_lloyd * (target - current_pts[i])
                
            new_pts = _normalize(new_pts)
            current_pts = new_pts
        else:
            # Greedy: Shrink the largest facet
            max_idx = np.argmax(radii)
            max_rad = radii[max_idx]
            
            if max_rad < 1e-6:
                break
                
            tri = hull.simplices[max_idx]
            c = centers[max_idx]
            
            moves = np.zeros(3)
            for idx in tri:
                v = current_pts[idx]
                # Project c onto tangent plane at v
                proj = np.dot(v, c) * c
                tangent = c - proj
                norm_t = np.linalg.norm(tangent)
                if norm_t > 1e-10:
                    moves += tangent / norm_t
                    
            move_norm = np.linalg.norm(moves)
            if move_norm < 1e-10:
                # If stuck, try a small random perturbation
                current_pts += np.random.normal(0, 0.001, (N, 3))
                current_pts = _normalize(current_pts)
            else:
                moves = moves / move_norm
                for idx in tri:
                    current_pts[idx] += step_size_greedy * moves
                    current_pts[idx] /= np.linalg.norm(current_pts[idx])
        
        # Update hull and radii every few iterations to save time
        if iteration % 5 == 0 or iteration == 1:
            try:
                hull = ConvexHull(current_pts)
                radii, centers = _compute_circumradii(current_pts, hull)
                new_cr, _, _ = _covering_radius(current_pts)
                
                if new_cr < best_radius - 1e-10:
                    best_radius = new_cr
                    best_pts = current_pts.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        # Apply repulsive force to break symmetry
                        rep = _repulsive_force(current_pts)
                        current_pts = current_pts + 0.01 * rep
                        current_pts = _normalize(current_pts)
                        
                        # Reduce step sizes
                        step_size_greedy *= 0.9
                        step_size_lloyd *= 0.9
                        no_improve_count = 0
                        if step_size_greedy < 1e-5:
                            break
            except Exception:
                # Hull failed, revert to best
                current_pts = best_pts.copy()
                break
        
        # Nelder-Mead Polish:
        # Every 20 iterations, if we have time, try to optimize the points 
        # associated with the top 3 largest facets using Nelder-Mead.
        if iteration % 20 == 0 and time.time() < deadline - 20.0:
            try:
                # Identify indices of points in the top 3 largest facets
                top_3 = np.argsort(radii)[-3:]
                top_facets = hull.simplices[top_3]
                top_point_indices = set()
                for tri in top_facets:
                    for idx in tri:
                        top_point_indices.add(idx)
                top_point_indices = list(top_point_indices)
                
                if len(top_point_indices) > 0 and len(top_point_indices) < N:
                    # Create a copy for optimization
                    opt_pts = current_pts.copy()
                    
                    def objective(x):
                        # Reconstruct points from x
                        x = x.reshape(-1, 3)
                        x = _normalize(x)
                        # Replace the optimized points
                        for i, idx in enumerate(top_point_indices):
                            x[idx] = opt_pts[idx]
                        # Compute covering radius
                        cr, _, _ = _covering_radius(x)
                        return cr
                    
                    # Initial guess
                    x0 = np.concatenate([current_pts[top_point_indices].flatten()])
                    
                    from scipy.optimize import minimize
                    result = minimize(objective, x0, method='Nelder-Mead', 
                                     options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
                    
                    if result.fun < best_radius:
                        # Update points
                        result_pts = _normalize(result.x.reshape(-1, 3))
                        for i, idx in enumerate(top_point_indices):
                            current_pts[idx] = result_pts[i]
                        current_pts = _normalize(current_pts)
                        
                        # Update hull
                        hull = ConvexHull(current_pts)
                        radii, centers = _compute_circumradii(current_pts, hull)
                        new_cr, _, _ = _covering_radius(current_pts)
                        
                        if new_cr < best_radius:
                            best_radius = new_cr
                            best_pts = current_pts.copy()
            except Exception:
                pass
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try multiple initializations and pick the best
    best_overall = None
    best_overall_radius = np.inf
    
    # Initialization 1: Fibonacci spiral with seed 42
    pts1 = _fibonacci_init(seed=42)
    r1, _, _ = _covering_radius(pts1)
    if r1 < best_overall_radius:
        best_overall_radius = r1
        best_overall = pts1.copy()
    
    # Initialization 2: Fibonacci spiral with seed 123
    pts2 = _fibonacci_init(seed=123)
    r2, _, _ = _covering_radius(pts2)
    if r2 < best_overall_radius:
        best_overall_radius = r2
        best_overall = pts2.copy()
    
    # Initialization 3: Fibonacci spiral with seed 456
    pts3 = _fibonacci_init(seed=456)
    r3, _, _ = _covering_radius(pts3)
    if r3 < best_overall_radius:
        best_overall_radius = r3
        best_overall = pts3.copy()
    
    # Optimize the best initialization
    try:
        optimized = _opti_cov(best_overall, deadline)
        if np.all(np.isfinite(optimized)):
            final_radius, _, _ = _covering_radius(optimized)
            if final_radius < best_overall_radius:
                best_overall = optimized
                best_overall_radius = final_radius
    except Exception:
        pass
    
    return _normalize(best_overall)
