"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _facet_angles(points):
    """
    Compute the covering angles (in radians) for each facet of the convex hull.
    Returns (angles_array, simplices_array, normals_array).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.array([np.pi]), np.array([[[0, 1, 2]]]), np.array([[1.0, 0.0, 0.0]])
    
    simplices = hull.simplices
    n_facets = len(simplices)
    angles = np.zeros(n_facets)
    normals = np.zeros((n_facets, 3))
    
    for k, simplex in enumerate(simplices):
        v1 = points[simplex[0]]
        v2 = points[simplex[1]]
        v3 = points[simplex[2]]
        
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            angles[k] = np.pi
            normals[k] = v1
            continue
        normal = normal / norm
        
        # Ensure normal points outward
        if np.dot(normal, v1) < 0:
            normal = -normal
            
        normals[k] = normal
        
        d1 = np.clip(np.dot(normal, v1), -1.0, 1.0)
        d2 = np.clip(np.dot(normal, v2), -1.0, 1.0)
        d3 = np.clip(np.dot(normal, v3), -1.0, 1.0)
        
        angles[k] = max(np.arccos(d1), np.arccos(d2), np.arccos(d3))
        
    return angles, simplices, normals

def _covering_radius_rad(points):
    """Compute the covering radius in radians."""
    angles, _, _ = _facet_angles(points)
    return np.max(angles)

def _covering_radius_deg(points):
    """Compute the covering radius in degrees."""
    return np.degrees(_covering_radius_rad(points))

def _fibonacci_spiral(n):
    """Generate n points on the unit sphere using a Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _electrostatic_refine(pts, iters=50, k=1.0, deadline=None):
    """
    Refine points using electrostatic repulsion (Lloyd-like on sphere).
    Move each point away from its neighbors to spread them out.
    """
    pts = pts.copy()
    for it in range(iters):
        if deadline is not None and time.time() > deadline:
            break
        disp = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                d = np.dot(pts[i], pts[j])
                # Repulsion stronger when closer
                rep = k / (1.0 - d + 1e-10)
                disp[i] += rep * diff
        # Normalize displacement and apply
        for i in range(N):
            d_norm = np.linalg.norm(disp[i])
            if d_norm > 1e-12:
                pts[i] += 0.01 * disp[i] / d_norm
        pts = _normalize(pts)
    return pts

def _smooth_surrogate(points, beta=30.0):
    """
    Compute a smooth surrogate for the covering radius using log-sum-exp.
    Returns (value, angles_array, simplices_array, normals_array).
    """
    angles, simplices, normals = _facet_angles(points)
    if len(angles) == 0:
        return np.pi, angles, simplices, normals
    
    max_ang = np.max(angles)
    exp_angles = np.exp(beta * (angles - max_ang))
    log_sum = max_ang + np.log(np.sum(exp_angles)) / beta
    return log_sum, angles, simplices, normals

def _surrogate_gradient(points, beta=30.0, eps=1e-4):
    """
    Compute gradient of the smooth surrogate using finite differences.
    """
    grad = np.zeros_like(points)
    current_val, _, _, _ = _smooth_surrogate(points, beta)
    
    for i in range(N):
        for j in range(3):
            pts_plus = points.copy()
            pts_plus[i, j] += eps
            pts_plus = _normalize(pts_plus)
            val_plus, _, _, _ = _smooth_surrogate(pts_plus, beta)
            
            pts_minus = points.copy()
            pts_minus[i, j] -= eps
            pts_minus = _normalize(pts_minus)
            val_minus, _, _, _ = _smooth_surrogate(pts_minus, beta)
            
            grad[i, j] = (val_plus - val_minus) / (2 * eps)
            
    return grad

def _gradient_descent(pts, max_iters=30, lr=0.02, beta=30.0, deadline=None):
    """
    Perform gradient descent on the smooth surrogate.
    """
    start_time = time.time()
    current_val, _, _, _ = _smooth_surrogate(pts, beta)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        grad = _surrogate_gradient(pts, beta)
        
        # Project to tangent space
        for i in range(N):
            radial = np.dot(grad[i], pts[i])
            grad[i] -= radial * pts[i]
            
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        grad = grad / grad_norm
        
        new_pts = pts - lr * grad
        new_pts = _normalize(new_pts)
        
        new_val, _, _, _ = _smooth_surrogate(new_pts, beta)
        
        if new_val < current_val:
            pts = new_pts
            current_val = new_val
        else:
            lr *= 0.7
            if lr < 1e-6:
                break
                
    return pts, current_val

def _greedy_facet_refine(pts, deadline, initial_step=0.05):
    """
    Greedy refinement: move vertices of the worst facet toward the facet normal.
    """
    pts = pts.copy()
    current_radius = _covering_radius_rad(pts)
    step_size = initial_step
    
    while time.time() < deadline:
        angles, simplices, normals = _facet_angles(pts)
        if len(angles) == 0:
            break
            
        worst_idx = np.argmax(angles)
        worst_ang = angles[worst_idx]
        
        if worst_ang < 1e-10:
            break
            
        simplex = simplices[worst_idx]
        normal = normals[worst_idx]
        
        v1 = pts[simplex[0]].copy()
        v2 = pts[simplex[1]].copy()
        v3 = pts[simplex[2]].copy()
        
        # Move vertices toward the normal (center of the hole)
        pts[simplex[0]] = v1 + step_size * normal
        pts[simplex[1]] = v2 + step_size * normal
        pts[simplex[2]] = v3 + step_size * normal
        pts = _normalize(pts)
        
        new_radius = _covering_radius_rad(pts)
        
        if new_radius < current_radius:
            current_radius = new_radius
            step_size *= 0.995
        else:
            # Revert
            pts[simplex[0]] = v1
            pts[simplex[1]] = v2
            pts[simplex[2]] = v3
            pts = _normalize(pts)
            step_size *= 0.85
            if step_size < 1e-6:
                break
                
    return pts, current_radius

def _simulated_annealing(pts, deadline, t0=0.1, t1=1e-4, cooling=0.995):
    """
    Simulated annealing to escape local minima.
    Randomly perturb points and accept worse solutions with some probability.
    """
    pts = pts.copy()
    current_radius = _covering_radius_rad(pts)
    temperature = t0
    
    while time.time() < deadline and temperature > t1:
        # Pick a random point and perturb it
        i = np.random.randint(N)
        # Random direction on the sphere
        rand_vec = np.random.randn(3)
        rand_vec = rand_vec / np.linalg.norm(rand_vec)
        # Remove radial component
        radial = np.dot(rand_vec, pts[i])
        rand_vec -= radial * pts[i]
        rand_vec_norm = np.linalg.norm(rand_vec)
        if rand_vec_norm < 1e-10:
            continue
        rand_vec = rand_vec / rand_vec_norm
        
        step = 0.05 * np.sqrt(temperature / t0)
        new_pts = pts.copy()
        new_pts[i] = pts[i] + step * rand_vec
        new_pts = _normalize(new_pts)
        
        new_radius = _covering_radius_rad(new_pts)
        delta = new_radius - current_radius
        
        if delta < 0 or np.random.random() < np.exp(-delta / temperature):
            pts = new_pts
            current_radius = new_radius
            
        temperature *= cooling
        
    return pts, current_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Initialize with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_radius = _covering_radius_rad(pts)
    
    # Phase 1: Electrostatic refinement to spread points
    if time.time() < deadline - 60:
        pts = _electrostatic_refine(pts, iters=30, k=0.5, deadline=deadline - 60)
        current_radius = _covering_radius_rad(pts)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
    
    # Phase 2: Gradient descent on smooth surrogate
    if time.time() < deadline - 40:
        pts, _ = _gradient_descent(pts, max_iters=20, lr=0.02, beta=30.0, deadline=deadline - 40)
        current_radius = _covering_radius_rad(pts)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
    
    # Phase 3: Greedy facet refinement
    if time.time() < deadline - 20:
        pts, current_radius = _greedy_facet_refine(pts, deadline=deadline - 20)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
    
    # Phase 4: Simulated annealing to escape local minima
    if time.time() < deadline - 5:
        np.random.seed(42)
        pts, current_radius = _simulated_annealing(pts, deadline=deadline - 5)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
    
    # Phase 5: Final greedy polish
    if time.time() < deadline - 1:
        pts, current_radius = _greedy_facet_refine(pts, deadline=deadline - 1, initial_step=0.02)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
    
    # Ensure validity: check no coincident points, not in hemisphere, etc.
    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
