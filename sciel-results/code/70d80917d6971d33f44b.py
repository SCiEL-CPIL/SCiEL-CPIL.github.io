"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius_fast(pts):
    """
    Compute the covering radius (in radians) using convex hull facets.
    Returns (radius, worst_facet_indices) for gradient computation.
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use sampling
        return _covering_radius_sample(pts), None
    
    max_radius = 0.0
    worst_simplex = None
    
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Circumcenter direction
        x = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
        norm_x = np.linalg.norm(x)
        
        if norm_x < 1e-12:
            continue
        
        x = x / norm_x
        
        # Angular distance from x to each vertex
        da = np.dot(x, a)
        db = np.dot(x, b)
        dc = np.dot(x, c)
        
        # The circumradius is the max angular distance (x should be equidistant, but numerically)
        # Actually, the covering radius contribution is arccos of the max dot product
        max_dot = max(da, db, dc)
        max_dot = np.clip(max_dot, -1.0, 1.0)
        r = np.arccos(max_dot)
        
        if r > max_radius:
            max_radius = r
            worst_simplex = simplex
    
    return max_radius, worst_simplex

def _covering_radius_sample(pts):
    """Fallback: sample the sphere to estimate covering radius."""
    n_theta = 100
    n_phi = 200
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    max_radius = 0.0
    for theta in thetas:
        st = np.sin(theta)
        ct = np.cos(theta)
        y_batch = np.column_stack([
            st * np.cos(phis),
            st * np.sin(phis),
            np.full_like(phis, ct)
        ])
        dots = pts @ y_batch.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dists = np.arccos(np.min(dots, axis=0))
        max_radius = max(max_radius, np.max(min_dists))
    return max_radius

def _objective(pts):
    """Compute the objective (covering radius in radians)."""
    radius, _ = _covering_radius_fast(pts)
    return radius

def _gradient_descent_sphere(pts, max_time=10.0):
    """
    Gradient descent on the sphere using the convex hull facet information.
    For the worst facet, push the three vertices apart.
    """
    pts = _to_sphere(pts.copy())
    best_pts = pts.copy()
    best_obj = _objective(pts)
    
    start = time.time()
    lr = 0.02
    no_improve = 0
    
    while time.time() - start < max_time:
        radius, worst_simplex = _covering_radius_fast(pts)
        
        if radius < best_obj:
            best_obj = radius
            best_pts = pts.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 30:
                break
        
        if worst_simplex is None:
            break
        
        # Get the three vertices of the worst facet
        a_idx, b_idx, c_idx = worst_simplex
        a = pts[a_idx]
        b = pts[b_idx]
        c = pts[c_idx]
        
        # Circumcenter direction
        x = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-12:
            break
        x = x / norm_x
        
        # The worst point is at x (or -x if the cap is on the other side)
        # We want to push a, b, c away from x to reduce the circumradius
        # Gradient: for each vertex, the direction that reduces arccos(dot(x, v))
        # d/dv arccos(dot(x,v)) = -x / sqrt(1 - dot(x,v)^2)
        # But we want to MINIMIZE the circumradius, so we move v in the direction
        # that INCREASES dot(x, v), i.e., toward x.
        # Wait, that's wrong. If we move v toward x, the angular distance decreases.
        # But we want to REDUCE the covering radius, which means moving v toward x.
        # However, this might increase other facets' radii.
        
        # Actually, to reduce the circumradius of this facet, we should move
        # the vertices AWAY from the circumcenter direction x (if x is the
        # empty side) or TOWARD x (if x is the filled side).
        
        # Let's check: is x on the "empty" side? Check if x is outside the hull.
        # The outward normal of the facet should point away from the hull center.
        # For a convex hull, the outward normal satisfies dot(n, p) <= dot(n, a) for all p.
        # So if dot(x, pts.mean()) < dot(x, a), then x is outward.
        
        center = np.mean(pts, axis=0)
        center = center / np.linalg.norm(center) if np.linalg.norm(center) > 1e-10 else np.zeros(3)
        
        # If x is on the outward side (empty cap side), we want to move vertices
        # toward x to reduce the circumradius. If x is inward, move away from x.
        if np.dot(x, center) < np.dot(x, a):
            # x is outward: move vertices toward x
            direction = 1
        else:
            # x is inward: move vertices away from x (i.e., toward -x)
            direction = -1
        
        # Compute gradients for the three vertices
        for idx, v in [(a_idx, a), (b_idx, b), (c_idx, c)]:
            dot_xv = np.dot(x, v)
            dot_xv = np.clip(dot_xv, -1.0, 1.0)
            sin_theta = np.sqrt(max(1.0 - dot_xv * dot_xv, 1e-12))
            # Gradient of arccos(dot(x,v)) w.r.t. v is -x/sin(theta)
            # To decrease the radius, move v in direction -gradient = x/sin(theta)
            grad = direction * x / sin_theta
            # Project to tangent space at v
            grad_tangent = grad - np.dot(grad, v) * v
            pts[idx] = pts[idx] + lr * grad_tangent
        
        pts = _to_sphere(pts)
    
    return best_pts, best_obj

def _simulated_annealing(pts, max_time=15.0):
    """
    Simulated annealing to find a good configuration.
    """
    pts = _to_sphere(pts.copy())
    current_obj = _objective(pts)
    best_pts = pts.copy()
    best_obj = current_obj
    
    start = time.time()
    temperature = 0.03
    cooling = 0.999
    step_size = 0.05
    
    while time.time() < max_time:
        # Propose a random perturbation
        noise = np.random.randn(N, 3) * step_size
        new_pts = _to_sphere(pts + noise)
        
        new_obj = _objective(new_pts)
        
        delta = new_obj - current_obj
        if delta < 0 or (temperature > 1e-6 and np.random.random() < np.exp(-delta / temperature)):
            pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
        
        temperature *= cooling
        if temperature < 0.001:
            temperature = 0.001
            step_size = max(step_size * 0.95, 0.01)
    
    return best_pts, best_obj

def _repulsion_optimize(pts, max_time=10.0):
    """
    Optimize using a repulsion-based approach that approximates the covering radius minimization.
    """
    pts = _to_sphere(pts.copy())
    best_pts = pts.copy()
    best_obj = _objective(pts)
    
    start = time.time()
    lr = 0.01
    
    while time.time() - start < max_time:
        # Compute all pairwise distances
        dots = pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        # For each point, compute the maximum distance to any other point
        # (excluding self)
        max_dists = np.zeros(N)
        for i in range(N):
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            max_dists[i] = np.max(dists[i, mask])
        
        # Force: each point is attracted toward the direction that reduces
        # its maximum distance to others. This is a heuristic.
        forces = np.zeros_like(pts)
        for i in range(N):
            # Find the farthest point from i
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            far_idx = np.argmax(dists[i, mask])
            far_idx = np.where(mask)[0][far_idx]
            
            # Move i toward the farthest point? No, that would reduce the gap.
            # Actually, we want to move i in the direction that reduces the
            # covering radius. A simple heuristic: move each point toward
            # the centroid of its k nearest neighbors.
            k = 5
            nearest = np.argsort(dists[i, mask])[k]
            nearest_indices = np.where(mask)[0][nearest]
            centroid = np.mean(pts[nearest_indices], axis=0)
            centroid = centroid / np.linalg.norm(centroid) if np.linalg.norm(centroid) > 1e-10 else np.zeros(3)
            
            # Move toward centroid
            force = centroid - pts[i]
            force = force - np.dot(force, pts[i]) * pts[i]  # project to tangent
            forces[i] = force
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms == 0] = 1.0
        forces = forces / force_norms
        
        new_pts = _to_sphere(pts + lr * forces)
        new_obj = _objective(new_pts)
        
        if new_obj < best_obj:
            best_obj = new_obj
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            lr *= 0.95
            if lr < 0.001:
                break
    
    return best_pts, best_obj

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    np.random.seed(42)
    
    best_pts = None
    best_obj = np.pi  # maximum possible
    
    deadline = _DEADLINE
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initial_configs.append(_to_sphere(fib_pts))
    
    # 2. Random
    for _ in range(3):
        rand_pts = np.random.randn(N, 3)
        initial_configs.append(_to_sphere(rand_pts))
    
    # 3. Based on icosahedral symmetry (approximate)
    # Use a grid on the sphere
    n_lat = 7
    n_lon = 8
    grid_pts = []
    for lat in range(n_lat):
        theta = np.pi * (lat + 0.5) / n_lat
        for lon in range(n_lon):
            phi = 2 * np.pi * lon / n_lon
            grid_pts.append([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
    grid_pts = np.array(grid_pts[:N])
    initial_configs.append(_to_sphere(grid_pts))
    
    # Optimize each initial configuration
    for init_idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 15:
            break
        
        remaining = deadline - time.time()
        
        # Phase 1: Repulsion optimization
        if remaining > 10:
            pts, obj = _repulsion_optimize(init_pts, max_time=min(remaining - 10, 15))
            if obj < best_obj:
                best_obj = obj
                best_pts = pts.copy()
        
        # Phase 2: Gradient descent on worst facet
        remaining = deadline - time.time()
        if remaining > 8:
            pts, obj = _gradient_descent_sphere(best_pts if best_pts is not None else init_pts, max_time=min(remaining - 8, 10))
            if obj < best_obj:
                best_obj = obj
                best_pts = pts.copy()
        
        # Phase 3: Simulated annealing
        remaining = deadline - time.time()
        if remaining > 12:
            pts, obj = _simulated_annealing(best_pts if best_pts is not None else init_pts, max_time=min(remaining - 12, 12))
            if obj < best_obj:
                best_obj = obj
                best_pts = pts.copy()
        
        # Phase 4: Final gradient polish
        remaining = deadline - time.time()
        if remaining > 5:
            pts, obj = _gradient_descent_sphere(best_pts, max_time=min(remaining - 5, 8))
            if obj < best_obj:
                best_obj = obj
                best_pts = pts.copy()
    
    # Final: one more round of SA + gradient from the best
    if best_pts is not None and time.time() < deadline - 10:
        remaining = deadline - time.time()
        pts, obj = _simulated_annealing(best_pts, max_time=min(remaining - 5, 10))
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
        
        remaining = deadline - time.time()
        if remaining > 5:
            pts, obj = _gradient_descent_sphere(best_pts, max_time=min(remaining - 5, 5))
            if obj < best_obj:
                best_obj = obj
                best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = _to_sphere(initial_configs[0])
    
    # Ensure all points are on the unit sphere
    best_pts = _to_sphere(best_pts)
    
    # Validate: check for coincident points
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 0.9999:
        best_pts = _to_sphere(best_pts + np.random.randn(N, 3) * 0.01)
        best_pts = _to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
